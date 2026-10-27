using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Threading.Tasks;
using System.IO;

public class CS_TestRotateBones : MonoBehaviour
{
	public static CS_TestRotateBones instance;
	public Transform[]	Bone		= new Transform[10];
	//public float[]		Min			= new float[14];
	//public float[]		Max			= new float[14];
	public float[]		MoveVelocity	= new float[14];
	public float[]		Goal		= new float[14];

	public bool[]			AxisX		= new bool[14];
	public bool[]			AxisY		= new bool[14];
	public bool[]			AxisZ		= new bool[14];
	private bool[]			MoveFlag	= new bool[14];
	private int[]			MoveVector	= new int[14];

	public int[]			JointInverse	= new int[14];
	public int[]			JointOffset		= new int[14];
	public float[] JointAngles = new float[14];

	private float BoneNum = 14;

	private float TimeStep = 0.02f;

	private float[]		_progress	= new float[14];

	// Playback support
	private Coroutine _playbackRoutine;
	public float playbackSampleInterval = 1.0f;
	[Min(0.01f)] public float playbackSpeed = 1.0f;
	private bool _isPlayingBack = false;
	public int PlaybackFrame { get; private set; }
	public int PlaybackFrameCount { get; private set; }

	private class PlaybackSample
	{
		public int Id;
		public double Time;
		public float[] Angles;
	}

	// Simple CSV pose storage: map sample index -> joint angles
	private System.Collections.Generic.Dictionary<int, float[]> _posesById = new System.Collections.Generic.Dictionary<int, float[]>();
	public bool ExternalControl { get; private set; }
	private int motionGeneration;

	public bool BeginExternalControl()
	{
		if (ExternalControl) return false;
		StopPlayback();
		motionGeneration++;
		System.Array.Clear(MoveFlag, 0, MoveFlag.Length);
		ExternalControl = true;
		return true;
	}

	public void EndExternalControl()
	{
		motionGeneration++;
		System.Array.Clear(MoveFlag, 0, MoveFlag.Length);
		ExternalControl = false;
	}

	public void ApplyExternalPose(float[] angles)
	{
		if (!ExternalControl || angles == null || angles.Length != 14) return;
		for (int i = 0; i < 14; i++)
			if (float.IsNaN(angles[i]) || float.IsInfinity(angles[i])) return;
		for (int i = 0; i < 14; i++)
			JointAngles[i] = CS_JointLimitDefinition.ClampGoal(i, angles[i]);
		ApplyJointAnglesToBones();
	}

	public void Awake()
	{
		if (instance == null)
		{
			instance = this;
		}
	}


	// Playback API
	[ContextMenu("Play Default Dataset")]
	public void PlayDefaultDataset()
	{
		string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "../../../../"));
		string csvPath = Path.Combine(projectRoot, "dataset/commu_pose_dataset", "angle_joints.csv");
		PlayCsv(csvPath);
	}

	public void PlayCsv(string csvFilePath)
	{
		if (ExternalControl) return;
		if (!Application.isPlaying)
		{
			Debug.LogWarning("Enter Play mode before starting CSV playback.");
			return;
		}
		List<PlaybackSample> samples;
		try { samples = ReadPlaybackCsv(csvFilePath); }
		catch (System.Exception exception)
		{
			Debug.LogError("Cannot play CSV: " + exception.Message);
			return;
		}
		StopPlayback();
		motionGeneration++;
		System.Array.Clear(MoveFlag, 0, MoveFlag.Length);
		PlaybackFrameCount = samples.Count;
		_playbackRoutine = StartCoroutine(PlaybackRoutine(samples));
	}

	// Minimal CSV loader for fixed-format CSV: index,image,14 joint columns
	public void LoadCsv(string csvFilePath)
	{
		_posesById.Clear();
		if (!File.Exists(csvFilePath))
		{
			Debug.LogWarning("CS_TestRotateBones.LoadCsv: CSV not found: " + csvFilePath);
			return;
		}

		string[] lines = File.ReadAllLines(csvFilePath);
		if (lines == null || lines.Length == 0)
		{
			Debug.LogWarning("CS_TestRotateBones.LoadCsv: CSV is empty: " + csvFilePath);
			return;
		}

		// determine whether the first line is a header (non-numeric first column)
		int startIndex = 0;
		{
			string firstLine = lines[0].Trim();
			if (!string.IsNullOrEmpty(firstLine))
			{
				string[] firstCols = firstLine.Split(',');
				int dummy;
				if (firstCols.Length == 0 || !int.TryParse(firstCols[0], out dummy))
				{
					startIndex = 1; // skip header
				}
			}
		}

		for (int i = startIndex; i < lines.Length; i++)
		{
			string line = lines[i].Trim();
			if (string.IsNullOrEmpty(line)) continue;
			string[] cols = line.Split(',');
			if (cols.Length < 2 + JointAngles.Length) continue;

			int id;
			if (!int.TryParse(cols[0], out id)) continue;

			float[] arr = new float[JointAngles.Length];
			for (int j = 0; j < JointAngles.Length; j++)
			{
				float v = 0f;
				float.TryParse(cols[2 + j], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out v);
				arr[j] = v;
			}

			_posesById[id] = arr;
		}

		Debug.Log($"CS_TestRotateBones.LoadCsv: loaded {_posesById.Count} poses from {csvFilePath}");
	}

	// Apply a loaded pose by its sample id. Returns true if applied.
	public bool ApplyPoseById(int sampleId)
	{
		if (ExternalControl) return false;
		float[] arr;
		if (!_posesById.TryGetValue(sampleId, out arr))
		{
			Debug.LogWarning("CS_TestRotateBones.ApplyPoseById: pose id not found: " + sampleId);
			return false;
		}

		for (int j = 0; j < JointAngles.Length; j++)
		{
			JointAngles[j] = arr[j];
			MoveFlag[j] = false; // stop any ongoing Move-driven animation
		}

		ApplyJointAnglesToBones();
		Debug.Log("CS_TestRotateBones.ApplyPoseById: applied pose " + sampleId);
		return true;
	}

	[ContextMenu("Stop Playback")]
	public void StopPlayback()
	{
		if (_playbackRoutine != null)
		{
			StopCoroutine(_playbackRoutine);
			_playbackRoutine = null;
		}
		_isPlayingBack = false;
	}

	public bool IsPlayingBack()
	{
		return _isPlayingBack;
	}

	private List<PlaybackSample> ReadPlaybackCsv(string csvFilePath)
	{
		string[] lines = File.ReadAllLines(csvFilePath);
		var samples = new List<PlaybackSample>();
		bool timed = false;
		bool firstRow = true;
		var culture = System.Globalization.CultureInfo.InvariantCulture;
		var style = System.Globalization.NumberStyles.Float;
		for (int li = 0; li < lines.Length; li++)
		{
			string line = lines[li].Trim();
			if (string.IsNullOrEmpty(line)) continue;
			string[] cols = line.Split(',');
			int id;
			if (firstRow && !int.TryParse(cols[0], out id))
			{
				timed = cols.Length > 1 && cols[1].Trim() == "timestamp_seconds";
				firstRow = false;
				continue;
			}
			firstRow = false;
			if (cols.Length < 2 + JointAngles.Length || !int.TryParse(cols[0], out id))
				throw new InvalidDataException($"Invalid pose CSV row {li + 1}.");
			double timestamp = samples.Count * System.Math.Max(0.001, playbackSampleInterval);
			if (timed && (!double.TryParse(cols[1], style, culture, out timestamp) ||
				double.IsNaN(timestamp) || double.IsInfinity(timestamp) || timestamp < 0))
				throw new InvalidDataException($"Invalid timestamp at row {li + 1}.");
			if (samples.Count > 0 && timestamp <= samples[samples.Count - 1].Time)
				throw new InvalidDataException($"Timestamps must increase at row {li + 1}.");
			float[] angles = new float[JointAngles.Length];
			for (int j = 0; j < JointAngles.Length; j++)
			{
				float value;
				if (!float.TryParse(cols[2 + j], style, culture, out value) || float.IsNaN(value) || float.IsInfinity(value))
					throw new InvalidDataException($"Invalid joint {j} at row {li + 1}.");
				angles[j] = value;
			}
			samples.Add(new PlaybackSample { Id = id, Time = timestamp, Angles = angles });
		}
		if (samples.Count == 0) throw new InvalidDataException("CSV contains no poses.");
		return samples;
	}

	private IEnumerator PlaybackRoutine(List<PlaybackSample> samples)
	{
		_isPlayingBack = true;
		int index = 0;
		int appliedIndex = -1;
		double start = Time.realtimeSinceStartupAsDouble;
		double first = samples[0].Time;
		double interval = samples.Count > 1
			? samples[samples.Count - 1].Time - samples[samples.Count - 2].Time
			: System.Math.Max(0.001, playbackSampleInterval);
		double end = samples[samples.Count - 1].Time - first + interval;
		Debug.Log($"CSV playback started: {samples.Count} samples, {end:F3}s, speed {playbackSpeed:F2}.");
		// Schedule against elapsed time, avoiding accumulated per-frame wait drift.
		while (true)
		{
			double elapsed = (Time.realtimeSinceStartupAsDouble - start) * Mathf.Max(0.01f, playbackSpeed);
			while (index + 1 < samples.Count && samples[index + 1].Time - first <= elapsed) index++;
			if (index != appliedIndex)
			{
				for (int j = 0; j < JointAngles.Length; j++)
					JointAngles[j] = CS_JointLimitDefinition.ClampGoal(j, samples[index].Angles[j]);
				ApplyJointAnglesToBones();
				PlaybackFrame = samples[index].Id;
				appliedIndex = index;
			}
			if (elapsed >= end) break;
			yield return null;
		}
		_isPlayingBack = false;
		_playbackRoutine = null;
		Debug.Log($"CSV playback completed: last frame {PlaybackFrame}.");
	}

	// Apply current JointAngles to Bone transforms immediately (copy of final part of FixedUpdate)
	public void ApplyJointAnglesToBones()
	{
		if (Bone == null || Bone.Length <= 1)
			return;

		float bodyYaw = CS_JointLimitDefinition.ClampGoal(0, JointAngles[0]);
		float bodyPitch = CS_JointLimitDefinition.ClampGoal(1, JointAngles[1]);
		float facePitch = CS_JointLimitDefinition.ClampGoal(6, JointAngles[6]);
		float faceYaw = CS_JointLimitDefinition.ClampGoal(7, JointAngles[7]);
		float faceRoll = CS_JointLimitDefinition.ClampGoal(8, JointAngles[8]);
		float leftArmPitch = CS_JointLimitDefinition.ClampGoal(4, JointAngles[4]);
		float leftArmRoll = CS_JointLimitDefinition.ClampGoal(5, JointAngles[5]);
		float rightArmPitch = CS_JointLimitDefinition.ClampGoal(2, JointAngles[2]);
		float rightArmRoll = CS_JointLimitDefinition.ClampGoal(3, JointAngles[3]);
		float mouth = CS_JointLimitDefinition.ClampGoal(13, JointAngles[13]);
		float eyePitch = CS_JointLimitDefinition.ClampGoal(9, JointAngles[9]);
		float rightEyeYaw = CS_JointLimitDefinition.ClampGoal(10, JointAngles[10]);
		float leftEyeYaw = CS_JointLimitDefinition.ClampGoal(11, JointAngles[11]);
		float eyelid = CS_JointLimitDefinition.ClampGoal(12, JointAngles[12]);

		Bone[1].localEulerAngles = new Vector3(bodyYaw, bodyPitch, 0);
		Bone[2].localEulerAngles = new Vector3(facePitch, faceYaw, faceRoll);
		Bone[3].localEulerAngles = new Vector3(leftArmPitch, 0, leftArmRoll);
		Bone[4].localEulerAngles = new Vector3(rightArmPitch, 0, rightArmRoll);
		Bone[5].localEulerAngles = new Vector3(mouth, 0, 0);
		Bone[6].localEulerAngles = new Vector3(eyePitch, leftEyeYaw, 0);
		Bone[7].localEulerAngles = new Vector3(eyePitch, rightEyeYaw, 0);
		Bone[8].localEulerAngles = new Vector3(eyelid, 0, 0);
		Bone[9].localEulerAngles = new Vector3(eyelid, 0, 0);
	}

	public void Start()
    {
//		Min = -Mathf.Abs(Min)
        for(int i = 0; i < 14; i++)
        {
			_progress[i] = 0f;
			MoveVector[i] = 1;
			//Min[i] = 0f;
			//Max[i] = 0f;
			JointAngles[i] = 0f;
        }
    }

	public async Task Gesture(float[] sleep_list, float[] velocity_list, int[] index_list, float[] goal_list)
    {

		if (ExternalControl || _isPlayingBack) return;
		int generation = motionGeneration;
		for (int i = 0; i < goal_list.Length; i++)
		{
			if (ExternalControl || _isPlayingBack || generation != motionGeneration) return;
			await OnDownButton(sleep_list[i], velocity_list[i], index_list[i], goal_list[i]);
		}

			

		/*
		await OnDownButton(0.0f, 60.0f, 6, 6.0f);
		await OnDownButton(0.5f, 60.0f, 6, -12.0f);
		await OnDownButton(0.5f, 60.0f, 6, 0.0f);
		*/
	}


	//public void OnDownButton(int index,float max,float min,int sleep)
	//public void OnDownButton(float sleep, float velocity, int index, float goal)
	private async Task OnDownButton(float sleep, float velocity, int index, float goal)
	{
		if (ExternalControl || _isPlayingBack || !CS_JointLimitDefinition.IsValidIndex(index))
			return;

		//Thread.Sleep((int)(sleep*1000));
		int generation = motionGeneration;
		await Task.Delay((int)(sleep * 1000));
		if (ExternalControl || _isPlayingBack || generation != motionGeneration) return;
		MoveVelocity[index] = velocity;
		Goal[index] = CS_JointLimitDefinition.ClampGoal(index, (goal + JointOffset[index]) * JointInverse[index]);
		//Goal[index] = goal;
		//Debug.Log(Goal[index]);
		MoveFlag[index] = true;

		/*
		float TargetAngle = 0f;
		if (AxisX[index])
			TargetAngle = Bone[index].localEulerAngles.x;
		if (AxisY[index])
			TargetAngle = Bone[index].localEulerAngles.y;
		if (AxisZ[index])
			TargetAngle = Bone[index].localEulerAngles.z;
		if (TargetAngle > 180)
			TargetAngle += -360;
		*/
		if (JointAngles[index] < Goal[index])
			MoveVector[index] = 1;
		else
			MoveVector[index] = -1;
		/*
		Max[index] = max;
		Min[index] = min;
		if (_progress[index] <= 0.01f)
		{
			if (Max[index] == 0)
				return;

			Quaternion q = Bone[index].localRotation;
			if( q.x == 0 && q.y == 0 && q.z== 0 )
			{
				if ( Min[index] == 0 )
					_progress[index] = 0.5f;
				else
					_progress[index] = 1.0f;
			}
		}
		*/

	}

	/*
    void Update()
    {
		for (int index = 0; index < 14; index++)
		{
			if (0 < _progress[index])
			{
				if (Bone == null)
					return;

				Quaternion q = Bone[index].localRotation;

				_progress[index] -= 0.02f;


				if (0 < _progress[index])
				{
					float a = 0;

					if (0.5f < _progress[index])
						a = Min[index] * Mathf.Abs(Mathf.Sin(Mathf.PI * 2 * _progress[index])) / 180.0f;
					else
						a = Max[index] * Mathf.Abs(Mathf.Sin(Mathf.PI * 2 * _progress[index])) / 180.0f;

					if (AxisX[index])
						q.x = a;

					if (AxisY[index])
						q.y = a;

					if (AxisZ[index])
						q.z = a;
				}
				else
				{
					_progress[index] = 0;
					q.x = 0;
					q.y = 0;
					q.z = 0;
				}
				Bone[index].localRotation = q;
			}
		}

	}
	*/

	void FixedUpdate()
	{
		if (ExternalControl || _isPlayingBack) return;
		int diff_cnt = 0;
		for (int index = 0; index < BoneNum; index++)
		{
			if (MoveFlag[index])
			{
				if (Bone == null)
					break;

				float Diff;

				Diff = MoveVector[index] * MoveVelocity[index] * TimeStep;
				//Debug.Log(Diff);

				JointAngles[index] += Diff;

				/*
				Vector3 AddVector = new Vector3(0f, 0f, 0f);



				if (AxisX[index])
                {
					AddVector.x = Diff;
					JointAngles[index] += Diff;

				}



				if (AxisY[index])
				{
					AddVector.y = Diff;
					JointAngles[index] += Diff;

				}

				if (AxisZ[index])
				{
					AddVector.z = Diff;
					JointAngles[index] += Diff;

				}
				*/

				//Bone[index].Rotate(AddVector);

				//Debug.Log(AddVector);
				//Bone[index].localEulerAngles = Bone[index].localEulerAngles + AddVector;


				/*
				float TargetAngle = 0f;

				if (AxisX[index])
					TargetAngle = Bone[index].localEulerAngles.x;
				if (AxisY[index])
					TargetAngle = Bone[index].localEulerAngles.y;
				if (AxisZ[index])
					TargetAngle = Bone[index].localEulerAngles.z;
				//Debug.Log(Bone[index].localEulerAngles.x -360);
				if (TargetAngle > 180)
					TargetAngle += -360;
				//Debug.Log(TargetAngle);

				//Debug.Log(Bone[index].localEulerAngles);
				*/

				if ((JointAngles[index] + Diff > Goal[index] && MoveVector[index] > 0) || (JointAngles[index] + Diff < Goal[index] && MoveVector[index] < 0))
				{
					MoveFlag[index] = false;
					/*
					// goal�Ƃ̂���𒲐�
					Vector3 GoalVector = Bone[index].localEulerAngles;
					if (AxisX[index])
						GoalVector.x = Goal[index];
					if (AxisY[index])
						GoalVector.y = Goal[index];
					if (AxisZ[index])
						GoalVector.z = Goal[index];
					//Bone[index].localEulerAngles = GoalVector;
					*/
					JointAngles[index] = CS_JointLimitDefinition.ClampGoal(index, Goal[index]);
				}

				//Debug.Log(JointAngles[index]);
			}
		}
		float bodyYaw = CS_JointLimitDefinition.ClampGoal(0, JointAngles[0]);
		float bodyPitch = CS_JointLimitDefinition.ClampGoal(1, JointAngles[1]);
		float facePitch = CS_JointLimitDefinition.ClampGoal(6, JointAngles[6]);
		float faceYaw = CS_JointLimitDefinition.ClampGoal(7, JointAngles[7]);
		float faceRoll = CS_JointLimitDefinition.ClampGoal(8, JointAngles[8]);
		float leftArmPitch = CS_JointLimitDefinition.ClampGoal(4, JointAngles[4]);
		float leftArmRoll = CS_JointLimitDefinition.ClampGoal(5, JointAngles[5]);
		float rightArmPitch = CS_JointLimitDefinition.ClampGoal(2, JointAngles[2]);
		float rightArmRoll = CS_JointLimitDefinition.ClampGoal(3, JointAngles[3]);
		float mouth = CS_JointLimitDefinition.ClampGoal(13, JointAngles[13]);
		float eyePitch = CS_JointLimitDefinition.ClampGoal(9, JointAngles[9]);
		float rightEyeYaw = CS_JointLimitDefinition.ClampGoal(10, JointAngles[10]);
		float leftEyeYaw = CS_JointLimitDefinition.ClampGoal(11, JointAngles[11]);
		float eyelid = CS_JointLimitDefinition.ClampGoal(12, JointAngles[12]);
			//Debug.Log(new Vector3(JointAngles[2], 0, JointAngles[3]));
			//Body
		diff_cnt += SearchDiff(Bone[1].localEulerAngles, new Vector3(bodyYaw, bodyPitch, 0));
		diff_cnt += SearchDiff(Bone[2].localEulerAngles, new Vector3(facePitch, faceYaw, faceRoll));
		diff_cnt += SearchDiff(Bone[3].localEulerAngles, new Vector3(leftArmPitch, 0, leftArmRoll));
		diff_cnt += SearchDiff(Bone[4].localEulerAngles, new Vector3(rightArmPitch, 0, rightArmRoll));
		diff_cnt += SearchDiff(Bone[5].localEulerAngles, new Vector3(mouth, 0, 0));
		diff_cnt += SearchDiff(Bone[6].localEulerAngles, new Vector3(eyePitch, leftEyeYaw, 0));
		diff_cnt += SearchDiff(Bone[7].localEulerAngles, new Vector3(eyePitch, rightEyeYaw, 0));
		diff_cnt += SearchDiff(Bone[8].localEulerAngles, new Vector3(eyelid, 0, 0));
		diff_cnt += SearchDiff(Bone[9].localEulerAngles, new Vector3(eyelid, 0, 0));
			if (CS_Operation.instance != null)
				CS_Operation.instance.Operation_Flag = diff_cnt != 0;
		
			
		Bone[1].localEulerAngles = new Vector3(bodyYaw, bodyPitch, 0);
		//Face
		Bone[2].localEulerAngles = new Vector3(facePitch, faceYaw, faceRoll);
		//Arm_L
		Bone[3].localEulerAngles = new Vector3(leftArmPitch, 0, leftArmRoll);
		//Arm_R
		Bone[4].localEulerAngles = new Vector3(rightArmPitch, 0, rightArmRoll);
		//Mouse
		Bone[5].localEulerAngles = new Vector3(mouth, 0, 0);
		//Eye_L(�����̉E��)
		Bone[6].localEulerAngles = new Vector3(eyePitch, leftEyeYaw, 0);
		//Eye_R(�����̍���)
		Bone[7].localEulerAngles = new Vector3(eyePitch, rightEyeYaw, 0);
		//Eyelid_L
		Bone[8].localEulerAngles = new Vector3(eyelid, 0, 0);
		//Eyelid_R
		Bone[9].localEulerAngles = new Vector3(eyelid, 0, 0);
	}

	public bool IsMoving()
	{
		for (int i = 0; i < MoveFlag.Length; i++)
		{
			if (MoveFlag[i]) return true;
		}
		return false;
	}

	int SearchDiff(Vector3 vec1, Vector3 vec2)
	{
		if (vec1 == vec2)
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}
	
}
