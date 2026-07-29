using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CS_CommU : MonoBehaviour
{
	public int FrameRate = 60;

	public Transform T_Base;
	public Transform T_Body;
	public Transform T_Face;
	public Transform T_Arm_L;
	public Transform T_Arm_R;
	public Transform T_Mouth;
	public Transform T_Eye_L;
	public Transform T_Eye_R;
	public Transform T_Eyelid_L;
	public Transform T_Eyelid_R;

	public Transform LookAtTarget = null;

	public bool enableLookAt = true;


	void Start()
    {
		T_Base	= gameObject.transform.GetChild(0);
		T_Body	= T_Base.transform.GetChild(0);
		T_Face	= T_Body.transform.GetChild(0);
		T_Arm_L	= T_Body.transform.GetChild(1);
		T_Arm_R = T_Body.transform.GetChild(2);
		T_Mouth = T_Face.transform.GetChild(0);
		T_Eye_L = T_Face.transform.GetChild(1);
		T_Eye_R = T_Face.transform.GetChild(2);
		T_Eyelid_L = T_Face.transform.GetChild(3);
		T_Eyelid_R = T_Face.transform.GetChild(4);

		Application.targetFrameRate = FrameRate;


/*
		Debug.Log(T_Base.name);
		Debug.Log(T_Body.name);
		Debug.Log(T_Face.name);
		Debug.Log(T_Arm_L.name);
		Debug.Log(T_Arm_R.name);

		Debug.Log(T_Mouth.name);
		Debug.Log(T_Eye_L.name);
		Debug.Log(T_Eye_R.name);
		Debug.Log(T_Eyelid_L.name);
		Debug.Log(T_Eyelid_R.name);
*/
	}

	void Update()
    {
		if(enableLookAt)
		{
			if (LookAtTarget == null)
				return;

			T_Eye_L.transform.LookAt( LookAtTarget);
			T_Eye_R.transform.LookAt( LookAtTarget);
		}

	}
}
