using System;
using UnityEngine;

public enum CommUJoint
{
	BodyYaw = 0,
	BodyPitch = 1,
	RightArmPitch = 2,
	RightArmRoll = 3,
	LeftArmPitch = 4,
	LeftArmRoll = 5,
	FacePitch = 6,
	FaceYaw = 7,
	FaceRoll = 8,
	EyePitch = 9,
	RightEyeYaw = 10,
	LeftEyeYaw = 11,
	Eyelid = 12,
	Mouth = 13,
}

[Serializable]
public struct CommUJointLimit
{
	public CommUJoint Joint;
	public Vector3 Axis;
	public float Min;
	public float Max;

	public CommUJointLimit(CommUJoint joint, Vector3 axis, float min, float max)
	{
		Joint = joint;
		Axis = axis;
		Min = min;
		Max = max;
	}
}

public static class CS_JointLimitDefinition
{
	public static readonly CommUJointLimit[] DefaultLimits =
	{
		new CommUJointLimit(CommUJoint.BodyYaw, new Vector3(0f, 1f, 0f), -20f, 20f),
		new CommUJointLimit(CommUJoint.BodyPitch, new Vector3(1f, 0f, 0f), -90f, 90f),
		new CommUJointLimit(CommUJoint.RightArmPitch, new Vector3(1f, 0f, 0f), -180f, 180f),
		new CommUJointLimit(CommUJoint.RightArmRoll, new Vector3(0f, 0f, 1f), -30f, 15f),
		new CommUJointLimit(CommUJoint.LeftArmPitch, new Vector3(1f, 0f, 0f), -180f, 180f),
		new CommUJointLimit(CommUJoint.LeftArmRoll, new Vector3(0f, 0f, 1f), -15f, 30f),
		new CommUJointLimit(CommUJoint.FacePitch, new Vector3(1f, 0f, 0f), -12f, 12f),
		new CommUJointLimit(CommUJoint.FaceYaw, new Vector3(0f, 1f, 0f), -0f, 0f),
		new CommUJointLimit(CommUJoint.FaceRoll, new Vector3(0f, 0f, 1f), -20f, 20f),
		new CommUJointLimit(CommUJoint.EyePitch, new Vector3(1f, 0f, 0f), -30f, 30f),
		new CommUJointLimit(CommUJoint.RightEyeYaw, new Vector3(0f, 1f, 0f), -30f, 30f),
		new CommUJointLimit(CommUJoint.LeftEyeYaw, new Vector3(0f, 1f, 0f), -30f, 30f),
		new CommUJointLimit(CommUJoint.Eyelid, new Vector3(1f, 0f, 0f), 0f, 45f),
		new CommUJointLimit(CommUJoint.Mouth, new Vector3(1f, 0f, 0f), -20f, 20f),
	};

	public static bool IsValidIndex(int index)
	{
		return index >= 0 && index < DefaultLimits.Length;
	}

	public static bool TryGetLimit(int index, out CommUJointLimit limit)
	{
		if (!IsValidIndex(index))
		{
			limit = default(CommUJointLimit);
			return false;
		}

		limit = DefaultLimits[index];
		return true;
	}

	public static CommUJointLimit GetLimit(CommUJoint joint)
	{
		return DefaultLimits[(int)joint];
	}

	public static bool TryGetLimit(CommUJoint joint, out CommUJointLimit limit)
	{
		return TryGetLimit((int)joint, out limit);
	}

	public static float ClampGoal(int index, float goal)
	{
		CommUJointLimit limit;
		if (!TryGetLimit(index, out limit))
		{
			return goal;
		}

		return Mathf.Clamp(goal, limit.Min, limit.Max);
	}
}
