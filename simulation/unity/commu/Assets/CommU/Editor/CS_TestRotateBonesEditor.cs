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

		serializedObject.ApplyModifiedProperties();
	}
}
