using UnityEngine;

public class DataCollect : DataCollectorBase
{
	private static readonly string[] JointColumnNames =
	{
		"body_yaw",
		"body_pitch",
		"right_arm_pitch",
		"right_arm_roll",
		"left_arm_pitch",
		"left_arm_roll",
		"face_pitch",
		"face_yaw",
		"face_roll",
		"eye_pitch",
		"right_eye_yaw",
		"left_eye_yaw",
		"eyelid",
		"mouth"
	};

	protected override string[] GetJointColumnNames()
	{
		return JointColumnNames;
	}
}