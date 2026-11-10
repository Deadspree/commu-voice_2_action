using System;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class ConversationSceneSetup
{
    public const string ScenePath = "Assets/Scenes/CommUConversation.unity";

    [MenuItem("CommU/Open Conversation Scene")]
    public static void Open()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Leave Play mode before opening the scene");
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) != null)
        { EditorSceneManager.OpenScene(ScenePath); return; }
        EditorSceneManager.OpenScene("Assets/Scenes/MediaPipeTracking.unity");
        foreach (var receiver in UnityEngine.Object.FindObjectsByType<CS_MediaPipeReceiver>(FindObjectsInactive.Include, FindObjectsSortMode.None)) receiver.enabled = false;
        foreach (var random in UnityEngine.Object.FindObjectsByType<CS_RandomMotion>(FindObjectsInactive.Include, FindObjectsSortMode.None)) random.enabled = false;
        foreach (var collector in UnityEngine.Object.FindObjectsByType<DataCollectorBase>(FindObjectsInactive.Include, FindObjectsSortMode.None)) collector.enabled = false;
        foreach (var operation in UnityEngine.Object.FindObjectsByType<CS_Operation>(FindObjectsInactive.Include, FindObjectsSortMode.None)) operation.enabled = false;
        foreach (var manual in UnityEngine.Object.FindObjectsByType<CommUSyudou>(FindObjectsInactive.Include, FindObjectsSortMode.None)) manual.enabled = false;
        var target = UnityEngine.Object.FindAnyObjectByType<CS_TestRotateBones>(FindObjectsInactive.Include);
        if (target == null) throw new InvalidOperationException("Source scene has no CommU controller");
        for (Transform parent = target.transform; parent != null; parent = parent.parent) parent.gameObject.SetActive(true);
        target.enabled = true;
        var root = new GameObject("CommUConversation");
        var state = root.AddComponent<ConversationState>();
        var player = root.AddComponent<SpeechMotionPlayer>();
        player.target = target;
        var microphone = root.AddComponent<MicrophoneCapture>();
        var client = root.AddComponent<ConversationClient>();
        client.state = state; client.player = player; client.microphone = microphone;
        root.AddComponent<ConversationUI>().client = client;
        var camera = Camera.main ?? UnityEngine.Object.FindAnyObjectByType<Camera>(FindObjectsInactive.Include);
        if (camera != null)
        {
            for (Transform parent = camera.transform; parent != null; parent = parent.parent) parent.gameObject.SetActive(true);
            camera.enabled = true;
            if (camera.GetComponent<AudioListener>() == null) camera.gameObject.AddComponent<AudioListener>();
        }
        EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), ScenePath);
        AssetDatabase.SaveAssets();
        Debug.Log("Created CommU conversation scene: " + ScenePath);
    }

    static void Check(bool condition, string message)
    { if (!condition) throw new Exception("Conversation validation failed: " + message); }

    [MenuItem("CommU/Validate Conversation")]
    public static void Validate()
    {
        Open();
        var client = UnityEngine.Object.FindAnyObjectByType<ConversationClient>();
        Check(client != null && client.player != null && client.microphone != null && client.state != null, "Scene references");
        Check(client.player.target != null && client.player.target.Bone.Skip(1).All(x => x != null), "Robot bone references");
        Check(!UnityEngine.Object.FindObjectsByType<CS_MediaPipeReceiver>(FindObjectsInactive.Include, FindObjectsSortMode.None).Any(x => x.enabled), "Competing receiver disabled");
        var message = new ConversationMessage { type = "response_chunk", turn_id = "validation", sequence = 0,
            sample_rate = 24000, sample_count = 4800, audio_pcm16 = ConversationProtocol.EncodeAudio(new float[4800]),
            joint_names = ConversationProtocol.JointNames, timestamps = new double[] { 0, .1 }, angles = new float[28] };
        message.angles[21] = 10;
        Check(ConversationProtocol.ValidateChunk(message), "Valid packet");
        Check(ConversationProtocol.DecodeAudio(message).Length == 4800, "PCM sample count");
        var pose = new float[14];
        ConversationProtocol.Interpolate(message, .05, pose);
        Check(Mathf.Abs(pose[7] - 5) < .001f, "Timestamp interpolation");
        message.timestamps[1] = 0;
        Check(!ConversationProtocol.ValidateChunk(message), "Duplicate timestamps rejected");
        message.timestamps[1] = .1;
        var temporary = new GameObject("Temporary conversation validation");
        try
        {
            var controller = temporary.AddComponent<CS_TestRotateBones>();
            controller.Bone = new Transform[10];
            for (int i = 0; i < 10; i++)
            { controller.Bone[i] = new GameObject("Bone " + i).transform; controller.Bone[i].SetParent(temporary.transform); }
            var player = temporary.AddComponent<SpeechMotionPlayer>();
            player.target = controller;
            player.Acquire();
            Check(controller.ExternalControl, "External pose ownership");
            player.Enqueue(message);
            Check(player.IsBusy, "Speech/motion queued together");
            player.StopAll();
            Check(!player.IsBusy, "Cancellation clears queued playback");
            pose[7] = 999;
            controller.ApplyExternalPose(pose);
            Check(controller.JointAngles[7] == CS_JointLimitDefinition.DefaultLimits[7].Max, "Existing joint clamp retained");
            player.ReleaseControl();
            Check(!controller.ExternalControl, "External control released");
        }
        finally { UnityEngine.Object.DestroyImmediate(temporary); }
        Debug.Log("CommU conversation validation PASSED (12 checks).");
    }

    public static void RunBatch()
    {
        try { Validate(); EditorApplication.Exit(0); }
        catch (Exception error) { Debug.LogException(error); EditorApplication.Exit(1); }
    }
}
