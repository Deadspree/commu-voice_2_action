using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CS_Target : MonoBehaviour
{
	private Vector3		_center;
	private Vector3		_pos;
	private float		_angle	= 0;
	public float		r		= 2;

	void Start()
    {
		_pos = _center = this.transform.position;
	}

    void Update()
    {
		_angle += 0.02f;
		_pos.x = _center.x + Mathf.Sin(_angle) * r * 2;
		_pos.y = _center.y + Mathf.Cos(_angle) * r;
		this.transform.position = _pos;
	}
}
