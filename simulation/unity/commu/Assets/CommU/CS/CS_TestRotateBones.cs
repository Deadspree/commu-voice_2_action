using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Threading.Tasks;

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
	public void Awake()
	{
		if (instance == null)
		{
			instance = this;
		}
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

	public async void Gesture(float[] sleep_list, float[] velocity_list, int[] index_list, float[] goal_list)
    {

		for (int i = 0; i < goal_list.Length; i++)
			await OnDownButton(sleep_list[i], velocity_list[i], index_list[i], goal_list[i]);

			

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
		if (!CS_JointLimitDefinition.IsValidIndex(index))
			return;

		//Thread.Sleep((int)(sleep*1000));
		await Task.Delay((int)(sleep * 1000));
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
		diff_cnt += SearchDiff(Bone[2].localEulerAngles, new Vector3(facePitch, faceRoll, faceYaw));
		diff_cnt += SearchDiff(Bone[3].localEulerAngles, new Vector3(leftArmPitch, 0, leftArmRoll));
		diff_cnt += SearchDiff(Bone[4].localEulerAngles, new Vector3(rightArmPitch, 0, rightArmRoll));
		diff_cnt += SearchDiff(Bone[5].localEulerAngles, new Vector3(mouth, 0, 0));
		diff_cnt += SearchDiff(Bone[6].localEulerAngles, new Vector3(eyePitch, leftEyeYaw, 0));
		diff_cnt += SearchDiff(Bone[7].localEulerAngles, new Vector3(eyePitch, rightEyeYaw, 0));
		diff_cnt += SearchDiff(Bone[8].localEulerAngles, new Vector3(eyelid, 0, 0));
		diff_cnt += SearchDiff(Bone[9].localEulerAngles, new Vector3(eyelid, 0, 0));
		if(diff_cnt == 0)
        {
			CS_Operation.instance.Operation_Flag = false;
        }
        else
        {
			CS_Operation.instance.Operation_Flag = true;
		}
		
			
		Bone[1].localEulerAngles = new Vector3(bodyYaw, bodyPitch, 0);
		//Face
		Bone[2].localEulerAngles = new Vector3(facePitch, faceRoll, faceYaw);
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
