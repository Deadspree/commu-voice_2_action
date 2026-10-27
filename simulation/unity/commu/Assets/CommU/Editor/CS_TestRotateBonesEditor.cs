using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(CS_TestRotateBones))]
public class CS_TestRotateBonesEditor : Editor
{
	private SerializedProperty boneProperty;
	private SerializedProperty moveVelocityProperty;
	private SerializedProperty goalProperty;
	private SerializedProperty jointAnglesProperty;
	private SerializedProperty jointOffsetProperty;
	private SerializedProperty jointInverseProperty;
	private SerializedProperty playbackSampleIntervalProperty;
	private SerializedProperty playbackSpeedProperty;
	private const string CsvPathPreference = "CS_TestRotateBones_CsvPath";
	private string csvPath;

	private void OnEnable()
	{
		boneProperty = serializedObject.FindProperty("Bone");
		moveVelocityProperty = serializedObject.FindProperty("MoveVelocity");
		goalProperty = serializedObject.FindProperty("Goal");
		jointAnglesProperty = serializedObject.FindProperty("JointAngles");
		jointOffsetProperty = serializedObject.FindProperty("JointOffset");
		jointInverseProperty = serializedObject.FindProperty("JointInverse");
		playbackSampleIntervalProperty = serializedObject.FindProperty("playbackSampleInterval");
		playbackSpeedProperty = serializedObject.FindProperty("playbackSpeed");
		csvPath = EditorPrefs.GetString(CsvPathPreference, "dataset/commu_pose_dataset/angle_joints.csv");
	}

	public override void OnInspectorGUI()
	{
		serializedObject.Update();

		EditorGUILayout.PropertyField(boneProperty, true);
		EditorGUILayout.PropertyField(moveVelocityProperty, true);
		EditorGUILayout.PropertyField(jointOffsetProperty, true);
		EditorGUILayout.PropertyField(jointInverseProperty, true);

		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Joint Sliders", EditorStyles.boldLabel);

		for (int i = 0; i < jointAnglesProperty.arraySize; i++)
		{
			SerializedProperty jointAngleProperty = jointAnglesProperty.GetArrayElementAtIndex(i);

			CommUJointLimit limit;
			float minValue = -180f;
			float maxValue = 180f;
			string label = ((CommUJoint)i).ToString();

			if (CS_JointLimitDefinition.TryGetLimit(i, out limit))
			{
				minValue = limit.Min;
				maxValue = limit.Max;
			}

			EditorGUILayout.BeginVertical(GUI.skin.box);
			EditorGUILayout.LabelField(label, EditorStyles.boldLabel);
			EditorGUILayout.Slider(jointAngleProperty, minValue, maxValue, new GUIContent("Joint Angle"));
			EditorGUILayout.EndVertical();
		}

		if (GUI.changed)
		{
			CS_TestRotateBones controller = (CS_TestRotateBones)target;
			EditorUtility.SetDirty(controller);
		}

		EditorGUILayout.Space();
		EditorGUILayout.LabelField("Pose Loader", EditorStyles.boldLabel);

		// reference to target
		CS_TestRotateBones ct = (CS_TestRotateBones)target;

		EditorGUI.BeginChangeCheck();
		csvPath = EditorGUILayout.TextField("CSV Path", csvPath);
		if (EditorGUI.EndChangeCheck())
			EditorPrefs.SetString(CsvPathPreference, csvPath);

		string projectRoot = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "../../../../"));
		string resolvedCsvPath = System.IO.Path.IsPathRooted(csvPath)
			? csvPath
			: System.IO.Path.Combine(projectRoot, csvPath);
		EditorGUILayout.HelpBox("Use an absolute path or a path relative to the repository root.", MessageType.Info);

		if (GUILayout.Button("Browse CSV..."))
		{
			string directory = System.IO.Path.GetDirectoryName(resolvedCsvPath);
			if (!System.IO.Directory.Exists(directory)) directory = projectRoot;
			string selectedPath = EditorUtility.OpenFilePanel("Select pose CSV", directory, "csv");
			if (!string.IsNullOrEmpty(selectedPath))
			{
				csvPath = selectedPath;
				EditorPrefs.SetString(CsvPathPreference, csvPath);
				resolvedCsvPath = selectedPath;
			}
		}

		using (new EditorGUI.DisabledScope(string.IsNullOrWhiteSpace(csvPath)))
		{
			if (GUILayout.Button("Load CSV"))
				ct.LoadCsv(resolvedCsvPath);
		}

		EditorGUILayout.Space();
		EditorGUILayout.LabelField("CSV Playback", EditorStyles.boldLabel);
		EditorGUILayout.PropertyField(playbackSpeedProperty, new GUIContent("Playback Speed"));
		EditorGUILayout.PropertyField(playbackSampleIntervalProperty, new GUIContent("Legacy CSV Interval (s)"));
		EditorGUILayout.HelpBox("Video CSVs replay their recorded timestamps. The interval applies only to older image-pose CSVs. Enter Play mode to play.", MessageType.Info);
		serializedObject.ApplyModifiedProperties();
		using (new EditorGUI.DisabledScope(!Application.isPlaying || ct.ExternalControl || string.IsNullOrWhiteSpace(csvPath)))
		{
			if (GUILayout.Button("Play CSV")) ct.PlayCsv(resolvedCsvPath);
		}
		using (new EditorGUI.DisabledScope(!ct.IsPlayingBack()))
		{
			if (GUILayout.Button("Stop CSV Playback")) ct.StopPlayback();
		}
		if (ct.ExternalControl)
			EditorGUILayout.HelpBox("Disable external tracking before playing a CSV.", MessageType.Warning);
		if (ct.IsPlayingBack())
		{
			EditorGUILayout.LabelField("Playing frame", ct.PlaybackFrame + " (" + ct.PlaybackFrameCount + " samples)");
			Repaint();
		}

		// apply-by-id UI
		 int applyId = EditorPrefs.GetInt("CS_TestRotateBones_ApplyId", 0);
		 applyId = EditorGUILayout.IntField("Apply Pose Id", applyId);
		 if (GUILayout.Button("Apply Pose By Id (Play Mode)"))
		 {
			 if (!Application.isPlaying)
			 {
				 Debug.LogWarning("Enter Play mode to apply poses by id.");
			 }
			 else
			 {
				 if (CS_TestRotateBones.instance == null)
				 {
					 Debug.LogWarning("CS_TestRotateBones.instance is null. Ensure the component exists in the scene.");
				 }
				 else
				 {
					 CS_TestRotateBones.instance.ApplyPoseById(applyId);
				 }
			 }
		 }
		EditorPrefs.SetInt("CS_TestRotateBones_ApplyId", applyId);

		serializedObject.ApplyModifiedProperties();
	}
}

// Keep the requested preview across the domain reload when entering Play mode.
[InitializeOnLoad]
public static class CS_VideoPosePreview
{
	private const string PendingCsv = "CommU.PendingVideoPoseCsv";

	static CS_VideoPosePreview()
	{
		EditorApplication.playModeStateChanged += state =>
		{
			if (state != PlayModeStateChange.EnteredPlayMode) return;
			string path = SessionState.GetString(PendingCsv, "");
			SessionState.EraseString(PendingCsv);
			if (!string.IsNullOrEmpty(path)) EditorApplication.delayCall += () => Play(path);
		};
	}

	[MenuItem("CommU/Preview Video Joint CSV...")]
	public static void ChooseCsv()
	{
		string root = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "../../../../"));
		string path = EditorUtility.OpenFilePanel("Select video joint predictions", System.IO.Path.Combine(root, "dataset/commu_pose_dataset/test_video"), "csv");
		if (string.IsNullOrEmpty(path)) return;
		EditorPrefs.SetString("CS_TestRotateBones_CsvPath", path);
		if (EditorApplication.isPlaying) Play(path);
		else
		{
			SessionState.SetString(PendingCsv, path);
			EditorApplication.isPlaying = true;
		}
	}

	private static void Play(string path)
	{
		if (!EditorApplication.isPlaying) return;
		var controller = Object.FindAnyObjectByType<CS_TestRotateBones>();
		if (controller == null)
		{
			Debug.LogError("No CS_TestRotateBones component in the active scene. Open the CommU simulation scene first.");
			return;
		}
		Selection.activeGameObject = controller.gameObject;
		if (controller.ExternalControl)
		{
			Debug.LogWarning("Disable external tracking before previewing a CSV.");
			return;
		}
		controller.PlayCsv(path);
	}
}
