using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CS_TestRotateBone : MonoBehaviour
{
	public Transform	Bone		= null;
	public float		Min			=-45.0f;
	public float		Max			= 45.0f;

	public bool			AxisX		= true;
	public bool			AxisY		= false;
	public bool			AxisZ		= false;


	private float		_progress	= 0;

	void Start()
    {
//		Min = -Mathf.Abs(Min)
        
    }



	public void OnDownButton()
	{
		if (_progress <= 0.01f)
		{
			if (Max == 0)
				return;

			Quaternion q = Bone.localRotation;
			if( q.x == 0 && q.y == 0 && q.z== 0 )
			{
				if ( Min == 0 )
					_progress = 0.5f;
				else
					_progress = 1.0f;
			}
		}

	}

    void Update()
    {
		if (0 < _progress)
		{
			if (Bone == null)
				return;

			Quaternion q = Bone.localRotation;

			_progress -= 0.02f;


			if (0 < _progress)
			{
				float a = 0;

				if (0.5f < _progress)
					a = Min * Mathf.Abs(Mathf.Sin(Mathf.PI *2* _progress)) / 180.0f;
				else
					a = Max * Mathf.Abs(Mathf.Sin(Mathf.PI *2* _progress)) / 180.0f;

				if (AxisX)
					q.x = a;

				if (AxisY)
					q.y = a;

				if (AxisZ)
					q.z = a;
			}
			else
			{
				_progress = 0;
				q.x = 0;
				q.y = 0;
				q.z = 0;
			}
			Bone.localRotation = q;
		}
	}
}
