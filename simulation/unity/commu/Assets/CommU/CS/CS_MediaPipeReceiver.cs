using System;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
using UnityEngine.InputSystem;
#endif

// Poll a bounded local UDP queue on the Unity thread; never mutate transforms off-thread.
[DefaultExecutionOrder(100)]
public class CS_MediaPipeReceiver : MonoBehaviour
{
    [Serializable]
    public class PosePacket
    {
        public int version;
        public string session;
        public long sequence;
        public double timestamp_ms;
        public float[] angles;
        public bool[] valid;
    }

    public CS_TestRotateBones target;
    public int port = 5055;
    public float disconnectSeconds = 0.6f;
    public float returnSeconds = 0.3f;
    [Header("PS4 Controller Teleoperation")]
    public bool enableControllerTeleop = true;
    [Range(0f, 0.5f)] public float controllerDeadzone = 0.15f;
    [Tooltip("Arm pitch movement speed at full stick deflection, in degrees per second.")]
    public float armPitchSpeed = 121.68f;
    [Tooltip("Arm roll movement speed at full stick deflection, in degrees per second.")]
    public float armRollSpeed = 46.8f;
    public float bodyYawSpeed = 50.7f;
    public float bodyPitchSpeed = 60.84f;
    [Tooltip("Write actual applied joint angles to a new CSV under Application.persistentDataPath.")]
    public bool recordAppliedAngles = false;
    public string status = "Not started";
    UdpClient socket;
    StreamWriter recording;
    CS_CommU robot;
    bool previousLookAt;
    bool ownsTarget;
    string session;
    long sequence = -1;
    double sourceTimestamp;
    float receivedAt;
    readonly float[] applied = new float[14];
    readonly bool[] validity = new bool[14];
    float manualBodyYaw;
    float manualBodyPitch;
    readonly float[] manualArmAngles = new float[4];
    static readonly string[] Columns = { "body_yaw", "body_pitch", "right_arm_pitch", "right_arm_roll",
        "left_arm_pitch", "left_arm_roll", "face_pitch", "face_yaw", "face_roll", "eye_pitch",
        "right_eye_yaw", "left_eye_yaw", "eyelid", "mouth" };

    void Start() { Connect(); }
    void OnEnable() { if (target != null && socket == null && Application.isPlaying) Connect(); }

    void Connect()
    {
        if (socket != null) return;
        if (target == null) target = FindAnyObjectByType<CS_TestRotateBones>();
        if (target == null || target.Bone == null || target.Bone.Length < 10)
        { Fail("Assign a CommU controller with all 10 bone references."); return; }
        for (int i = 1; i < 10; i++)
            if (target.Bone[i] == null) { Fail("CommU has a missing bone reference."); return; }
        try
        {
            socket = new UdpClient(new IPEndPoint(IPAddress.Loopback, port));
            socket.Client.Blocking = false;
            if (!target.BeginExternalControl())
            { Fail("Another receiver already owns this CommU controller."); return; }
            ownsTarget = true;
            robot = target.GetComponentInParent<CS_CommU>();
            if (robot != null) { previousLookAt = robot.enableLookAt; robot.enableLookAt = false; }
            Array.Copy(target.JointAngles, applied, 14);
            manualBodyYaw = CS_JointLimitDefinition.ClampGoal(0, target.JointAngles[0]);
            manualBodyPitch = CS_JointLimitDefinition.ClampGoal(1, target.JointAngles[1]);
            manualArmAngles[0] = CS_JointLimitDefinition.ClampGoal(4, target.JointAngles[4]);
            manualArmAngles[1] = CS_JointLimitDefinition.ClampGoal(5, target.JointAngles[5]);
            manualArmAngles[2] = CS_JointLimitDefinition.ClampGoal(2, target.JointAngles[2]);
            manualArmAngles[3] = CS_JointLimitDefinition.ClampGoal(3, target.JointAngles[3]);
            session = null;
            sequence = -1;
            receivedAt = Time.unscaledTime;
            if (recordAppliedAngles)
            {
                string path = Path.Combine(Application.persistentDataPath,
                    "commu-motion-" + DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff") + ".csv");
                recording = new StreamWriter(new FileStream(path, FileMode.CreateNew, FileAccess.Write));
                recording.WriteLine("session,sequence,source_timestamp_ms,unity_time,stale,valid_mask," + string.Join(",", Columns));
                Debug.Log("CommU applied motion recording: " + path);
            }
            status = "Waiting on 127.0.0.1:" + port;
        }
        catch (Exception ex) { Fail(ex.Message); }
    }

    void Fail(string message)
    {
        status = message;
        Debug.LogError("CommU tracking: " + message, this);
        Disconnect();
        enabled = false;
    }

    public static bool IsValidPacket(PosePacket packet)
    {
        if (packet == null || packet.version != 1 || !Guid.TryParse(packet.session, out _) ||
            packet.sequence < 0 || double.IsNaN(packet.timestamp_ms) || double.IsInfinity(packet.timestamp_ms) ||
            packet.timestamp_ms < 0 || packet.angles == null || packet.angles.Length != 14 ||
            packet.valid == null || packet.valid.Length != 14) return false;
        foreach (float angle in packet.angles)
            if (float.IsNaN(angle) || float.IsInfinity(angle)) return false;
        return true;
    }

    void LateUpdate()
    {
        if (socket == null || !ownsTarget) return;
        try
        {
            // Drain normally; cap work so a broken sender cannot freeze a frame.
            for (int i = 0; i < 128 && socket.Available > 0; i++)
            {
                IPEndPoint sender = null;
                byte[] bytes = socket.Receive(ref sender);
                if (bytes.Length > 8192) continue;
                PosePacket packet;
                try { packet = JsonUtility.FromJson<PosePacket>(Encoding.UTF8.GetString(bytes)); }
                catch (ArgumentException) { continue; }
                if (!IsValidPacket(packet)) continue;
                if (packet.session == session && (packet.sequence <= sequence || packet.timestamp_ms <= sourceTimestamp)) continue;
                session = packet.session;
                sequence = packet.sequence;
                sourceTimestamp = packet.timestamp_ms;
                receivedAt = Time.unscaledTime;
                for (int j = 0; j < 14; j++)
                {
                    applied[j] = CS_JointLimitDefinition.ClampGoal(j, packet.angles[j]);
                    validity[j] = packet.valid[j];
                }
                status = "Receiving frame " + sequence;
            }
            bool stale = Time.unscaledTime - receivedAt > Mathf.Max(0.05f, disconnectSeconds);
            if (stale)
            {
                status = "Tracking disconnected: returning to neutral";
                float alpha = 1f - Mathf.Exp(-Time.unscaledDeltaTime / Mathf.Max(0.001f, returnSeconds));
                for (int j = 0; j < 14; j++) applied[j] = Mathf.Lerp(applied[j], 0f, alpha);
                Array.Clear(validity, 0, validity.Length);
            }
            ApplyControllerTeleop();
            target.ApplyExternalPose(applied);
            if (recording != null && session != null)
            {
                var row = new StringBuilder();
                row.Append(session).Append(',').Append(sequence).Append(',')
                    .Append(sourceTimestamp.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(Time.unscaledTime.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(stale ? "1" : "0").Append(',');
                for (int j = 0; j < 14; j++) row.Append(validity[j] ? '1' : '0');
                foreach (float angle in target.JointAngles) row.Append(',').Append(angle.ToString("R", CultureInfo.InvariantCulture));
                recording.WriteLine(row.ToString());
            }
        }
        catch (SocketException ex)
        { if (ex.SocketErrorCode != SocketError.WouldBlock) Fail(ex.Message); }
        catch (IOException ex) { Fail(ex.Message); }
    }

    void ApplyControllerTeleop()
    {
        if (!enableControllerTeleop) return;
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
        Gamepad pad = Gamepad.current;
        if (pad == null) return;

        // D-pad directions move the torso at a fixed rate; release holds the angle.
        float yaw = (pad.rightShoulder.isPressed ? 1f : 0f) - (pad.rightTrigger.isPressed ? 1f : 0f);
        float pitch = (pad.dpad.left.isPressed ? 1f : 0f) - (pad.dpad.right.isPressed ? 1f : 0f);
        manualBodyYaw = CS_JointLimitDefinition.ClampGoal(0,
            manualBodyYaw + yaw * Mathf.Max(0f, bodyYawSpeed) * Time.unscaledDeltaTime);
        manualBodyPitch = CS_JointLimitDefinition.ClampGoal(1,
            manualBodyPitch + pitch * Mathf.Max(0f, bodyPitchSpeed) * Time.unscaledDeltaTime);
        applied[0] = manualBodyYaw;
        applied[1] = manualBodyPitch;

        Vector2 left = ApplyRadialDeadzone(pad.leftStick.ReadValue(), controllerDeadzone);
        Vector2 right = ApplyRadialDeadzone(pad.rightStick.ReadValue(), controllerDeadzone);
        float deltaTime = Time.unscaledDeltaTime;
        manualArmAngles[0] = CS_JointLimitDefinition.ClampGoal(4,
            manualArmAngles[0] + left.y * Mathf.Max(0f, armPitchSpeed) * deltaTime);
        manualArmAngles[1] = CS_JointLimitDefinition.ClampGoal(5,
            manualArmAngles[1] - left.x * Mathf.Max(0f, armRollSpeed) * deltaTime);
        manualArmAngles[2] = CS_JointLimitDefinition.ClampGoal(2,
            manualArmAngles[2] + right.y * Mathf.Max(0f, armPitchSpeed) * deltaTime);
        manualArmAngles[3] = CS_JointLimitDefinition.ClampGoal(3,
            manualArmAngles[3] - right.x * Mathf.Max(0f, armRollSpeed) * deltaTime);
        applied[4] = manualArmAngles[0];
        applied[5] = manualArmAngles[1];
        applied[2] = manualArmAngles[2];
        applied[3] = manualArmAngles[3];
#endif
    }

#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
    static Vector2 ApplyRadialDeadzone(Vector2 value, float deadzone)
    {
        float magnitude = value.magnitude;
        if (magnitude <= deadzone) return Vector2.zero;
        return value / magnitude * Mathf.Clamp01((magnitude - deadzone) / Mathf.Max(0.001f, 1f - deadzone));
    }

#endif

    void OnDisable() { Disconnect(); }
    void OnDestroy() { Disconnect(); }
    void Disconnect()
    {
        socket?.Close(); socket = null;
        recording?.Dispose(); recording = null;
        if (ownsTarget)
        {
            if (target != null) target.EndExternalControl();
            if (robot != null) robot.enableLookAt = previousLookAt;
        }
        ownsTarget = false;
    }
}
