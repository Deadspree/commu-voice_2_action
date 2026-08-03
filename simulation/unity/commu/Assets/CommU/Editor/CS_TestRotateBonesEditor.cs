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

	private void OnEnable()
	{
		boneProperty = serializedObject.FindProperty("Bone");
		moveVelocityProperty = serializedObject.FindProperty("MoveVelocity");
		goalProperty = serializedObject.FindProperty("Goal");
		jointAnglesProperty = serializedObject.FindProperty("JointAngles");
		jointOffsetProperty = serializedObject.FindProperty("JointOffset");
		jointInverseProperty = serializedObject.FindProperty("JointInverse");
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

		if (GUILayout.Button("Load CSV (project dataset/commu_pose_dataset/angle_joints.csv)"))
		{
			string projectRoot = System.IO.Path.GetFullPath(System.IO.Path.Combine(Application.dataPath, "../../../../"));
			string csvPath = System.IO.Path.Combine(projectRoot, "dataset/commu_pose_dataset", "angle_joints.csv");
			ct.LoadCsv(csvPath);
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
