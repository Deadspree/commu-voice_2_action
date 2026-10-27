import math
from pathlib import Path
import re
import numpy as np
import pytest
from motion_tracking.retarget import Config, Retargeter, LIMITS, JOINTS, rotation_angles


def body():
    # In MediaPipe coordinates, subject left is image right.
    p = np.zeros((33,3))
    p[11],p[12],p[23],p[24] = [.3,-.5,0],[-.3,-.5,0],[.2,.3,0],[-.2,.3,0]
    p[13],p[14] = [.3,0,0],[-.3,0,0]
    return p


def fast(**kw):
    return Retargeter(Config(calibration_frames=1, smoothing_seconds=0, expression_smoothing_seconds=0, **kw))


def rotation_y(deg):
    a=math.radians(deg)
    return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])


def test_neutral_and_missing_tracking():
    m=fast()
    a,v=m.update(0,body(),face_matrix=np.eye(4))
    assert np.allclose(a,0)
    assert v[:9].all()
    assert not v[9:].any()


def test_body_rotation_does_not_double_head_rotation():
    m=fast(head_reference="torso")
    m.update(0,body(),face_matrix=np.eye(4))
    r=rotation_y(30)
    converted=body()*[1,-1,-1]
    rotated=(converted@r.T)*[1,-1,-1]
    f=np.eye(4); f[:3,:3]=r
    a,v=m.update(.1,rotated,face_matrix=f)
    assert a[0]==pytest.approx(15)
    assert np.allclose(a[6:9],0,atol=1e-8)


def test_head_relative_to_body_and_clamped():
    m=fast(head_reference="torso");m.update(0,body(),face_matrix=np.eye(4))
    f=np.eye(4);f[:3,:3]=rotation_y(60)
    a,_=m.update(.1,body(),face_matrix=f)
    assert a[7]==pytest.approx(10)


def test_arms_are_independent():
    m=fast();p=body();p[13]=[.3,-1,0]
    a,v=m.update(0,p)
    assert a[4]==pytest.approx(-180)
    assert a[2]==pytest.approx(0)


def test_blink_mouth_hold_then_neutral():
    m=fast()
    a,_=m.update(0,blends={'eyeBlinkLeft':1,'eyeBlinkRight':1,'jawOpen':1})
    assert a[12:14].tolist()==[45,20]
    a,v=m.update(.2)
    assert a[12:14].tolist()==[45,20]
    assert not v.any()
    a,_=m.update(.5)
    assert np.allclose(a,0)


def test_occlusion_keeps_face_but_invalidates_body_and_head():
    m=fast(head_reference="torso");visibility=np.ones(33);visibility[23]=0
    _,v=m.update(0,body(),visibility,np.eye(4),{'jawOpen':.5})
    assert not v[:9].any()
    assert v[13]


def test_nan_and_degenerate_landmarks_are_not_valid():
    m=fast();p=body();p[11]=np.nan
    a,v=m.update(0,p,face_matrix=np.full((4,4),np.nan),blends={'jawOpen':float('nan')})
    assert np.isfinite(a).all()
    assert not v.any()
    _,v=m.update(.1,np.zeros((33,3)))
    assert not v.any()


def test_calibration_and_reset():
    m=Retargeter(Config(calibration_frames=2))
    _,v=m.update(0,body(),face_matrix=np.eye(4))
    assert not v[:2].any() and not v[6:9].any()
    m.update(.1,body(),face_matrix=np.eye(4))
    assert m.calibrated('head')
    m.reset()
    assert not m.calibrated('head')


def test_smoothing_and_timestamp_validation():
    m=Retargeter()
    a,_=m.update(0,blends={'jawOpen':1})
    assert 0<a[13]<40
    with pytest.raises(ValueError):m.update(0)
    with pytest.raises(ValueError):m.update(float('nan'))


def test_config_validation():
    with pytest.raises(ValueError):Config(gains=[1])
    with pytest.raises(ValueError):Config(hold_seconds=-1)


def test_limits_match_unity_source():
    path=Path(__file__).resolve().parents[2]/'simulation/unity/commu/Assets/CommU/CS/CS_JointLimitDefinition.cs'
    values=re.findall(r'new CommUJointLimit\(CommUJoint\.(\w+), new Vector3\([^)]*\), (-?\d+)f, (-?\d+)f\)',path.read_text())
    assert len(values)==len(JOINTS)
    assert np.array_equal(np.array([[float(lo),float(hi)] for _,lo,hi in values]),LIMITS)


@pytest.mark.parametrize("axis,index,gain", [(0,6,.4),(1,7,.3),(2,8,.7)])
def test_head_moves_without_any_body_landmarks(axis,index,gain):
    m=fast()
    m.update(0,face_matrix=np.eye(4))
    angle=math.radians(20)
    r=np.eye(4)
    first,second=((1,2),(2,0),(0,1))[axis]
    r[first,first]=r[second,second]=math.cos(angle)
    r[first,second]=-math.sin(angle)
    r[second,first]=math.sin(angle)
    a,v=m.update(.1,face_matrix=r)
    assert v[6:9].all()
    assert a[index]==pytest.approx(20*gain)
    assert m.head_status=="Tracking"
    assert not v[:6].any()


def test_head_continues_when_torso_disappears():
    m=fast()
    m.update(0,body(),face_matrix=np.eye(4))
    r=np.eye(4);r[:3,:3]=rotation_y(20)
    a,v=m.update(.1,face_matrix=r)
    assert a[7]==pytest.approx(6)
    assert v[6:9].all()


def test_calibrated_does_not_mean_still_tracking():
    m=fast();m.update(0,face_matrix=np.eye(4))
    _,v=m.update(.1)
    assert m.calibrated('head')
    assert not v[6:9].any()
    assert m.head_status=="Face lost"


def test_head_calibration_counter_without_body():
    m=Retargeter(Config(calibration_frames=2))
    m.update(0,face_matrix=np.eye(4))
    assert m.head_status=="Calibrating 1/2"
    m.update(.1,face_matrix=np.eye(4))
    assert m.head_status=="Tracking"
    m.reset()
    assert m.head_status=="Waiting for face"



def test_moderate_blink_score_reaches_joint_limit():
    m=fast()
    a,v=m.update(0,blends={'eyeBlinkLeft':.6,'eyeBlinkRight':.6})
    assert v[12]
    assert a[12]==pytest.approx(45)
    assert m.blink_scores==(.6,.6)


def test_short_blink_is_held_then_released():
    m=fast()
    m.update(0,blends={'eyeBlinkLeft':.8,'eyeBlinkRight':.8})
    opened={'eyeBlinkLeft':0,'eyeBlinkRight':0}
    a,_=m.update(.05,blends=opened)
    assert a[12]==pytest.approx(45)
    a,_=m.update(.15,blends=opened)
    assert a[12]==pytest.approx(0)


def test_open_eye_noise_and_partial_closure():
    m=fast()
    a,_=m.update(0,blends={'eyeBlinkLeft':.08,'eyeBlinkRight':.08})
    assert a[12]==0
    a,_=m.update(.1,blends={'eyeBlinkLeft':.325,'eyeBlinkRight':.325})
    assert a[12]==pytest.approx(15)


def test_blink_hold_reset_and_invalid_scores():
    m=fast()
    m.update(0,blends={'eyeBlinkLeft':1,'eyeBlinkRight':1})
    m.reset()
    a,_=m.update(0,blends={'eyeBlinkLeft':0,'eyeBlinkRight':0})
    assert a[12]==0
    _,v=m.update(.1,blends={'eyeBlinkLeft':float('nan'),'eyeBlinkRight':1})
    assert not v[12]
    assert m.blink_scores is None
    with pytest.raises(ValueError):Config(blink_open_score=.8,blink_closed_score=.2)
    with pytest.raises(ValueError):Config(blink_hold_seconds=-1)
