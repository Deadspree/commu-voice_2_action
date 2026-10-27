using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class CS_MediaPipeValidation
{
    [MenuItem("CommU/Validate MediaPipe Receiver")]
    public static void Run()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Run validation outside Play mode.");
        var root = new GameObject("Temporary tracking validation");
        try
        {
            var controller = root.AddComponent<CS_TestRotateBones>();
            controller.Bone = new Transform[10];
            for (int i = 0; i < 10; i++)
            {
                controller.Bone[i] = new GameObject("Bone " + i).transform;
                controller.Bone[i].SetParent(root.transform);
            }
            var receiver = root.AddComponent<CS_MediaPipeReceiver>();
            receiver.target = controller;
            // Port 0 requests an OS-assigned ephemeral port for the test.
            receiver.port = 0;
            Invoke(receiver, "Start");
            Check(controller.ExternalControl, "Receiver acquires controller");
            var field = typeof(CS_MediaPipeReceiver).GetField("socket", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var listener = (UdpClient)field.GetValue(receiver);
            int port = ((IPEndPoint)listener.Client.LocalEndPoint).Port;
            var packet = new CS_MediaPipeReceiver.PosePacket { version = 1, session = Guid.NewGuid().ToString(),
                sequence = 1, timestamp_ms = 1, angles = new float[14], valid = new bool[14] };
            packet.angles[7] = 999;
            using (var sender = new UdpClient())
            {
                Send(sender, listener, port, JsonUtility.ToJson(packet));
                Invoke(receiver, "LateUpdate");
                Check(controller.JointAngles[7] == 10, "Receiver clamps joint angle");
                Check(Mathf.Abs(Mathf.DeltaAngle(controller.Bone[2].localEulerAngles.y, 10)) < .01f, "Receiver applies face yaw on the Y axis");
                packet.angles[7] = -10;
                Send(sender, listener, port, JsonUtility.ToJson(packet));
                Invoke(receiver, "LateUpdate");
                Check(controller.JointAngles[7] == 10, "Duplicate packet rejected");
                packet.sequence = 2; packet.timestamp_ms = 2;
                packet.angles[12] = 45;
                Send(sender, listener, port, JsonUtility.ToJson(packet));
                Invoke(receiver, "LateUpdate");
                Check(controller.JointAngles[7] == -10, "New packet accepted");
                Check(Mathf.Abs(Mathf.DeltaAngle(controller.Bone[8].localEulerAngles.x, 45)) < .01f &&
                      Mathf.Abs(Mathf.DeltaAngle(controller.Bone[9].localEulerAngles.x, 45)) < .01f,
                      "Both eyelids close through UDP");
                packet.angles[12] = 0;
                packet.session = Guid.NewGuid().ToString(); packet.sequence = 0; packet.timestamp_ms = 0; packet.angles[7] = 5;
                Send(sender, listener, port, JsonUtility.ToJson(packet));
                Invoke(receiver, "LateUpdate");
                Check(controller.JointAngles[7] == 5, "Sender restart accepted");
                Check(Mathf.Abs(Mathf.DeltaAngle(controller.Bone[8].localEulerAngles.x, 0)) < .01f &&
                      Mathf.Abs(Mathf.DeltaAngle(controller.Bone[9].localEulerAngles.x, 0)) < .01f,
                      "Both eyelids reopen through UDP");
                Send(sender, listener, port, "{}");
                Invoke(receiver, "LateUpdate");
                Check(controller.JointAngles[7] == 5, "Malformed packet ignored");
            }
            packet.angles[0] = float.NaN;
            Check(!CS_MediaPipeReceiver.IsValidPacket(packet), "NaN rejected");
            Invoke(receiver, "OnDisable");
            Check(!controller.ExternalControl, "Controller released");
            Debug.Log("CommU MediaPipe receiver validation PASSED (11 checks).");
        }
        finally { UnityEngine.Object.DestroyImmediate(root); }
    }

    static void Invoke(CS_MediaPipeReceiver receiver, string method)
    {
        typeof(CS_MediaPipeReceiver).GetMethod(method,
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).Invoke(receiver, null);
    }

    static void Send(UdpClient client, UdpClient listener, int port, string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        client.Send(bytes, bytes.Length, new IPEndPoint(IPAddress.Loopback, port));
        Check(listener.Client.Poll(1000000, SelectMode.SelectRead), "Local UDP delivery within one second");
    }
    static void Check(bool value, string description)
    { if (!value) throw new Exception("Tracking validation failed: " + description); }

    public static void RunBatch()
    {
        try { Run(); EditorApplication.Exit(0); }
        catch (Exception ex) { Debug.LogException(ex); EditorApplication.Exit(1); }
    }
}
