using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public static class VideoComparisonSetup
{
    public const string ScenePath = "Assets/Scenes/VideoPoseComparison.unity";
    public const string DefaultRun = "dataset/lucy/generation_runs/speech2gesture/00271_6f35a901156dd4cd1fdf";

    [MenuItem("CommU/Open Video Comparison")]
    public static void Open()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Leave Play Mode before opening comparison");
        if (!Application.isBatchMode && !EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (File.Exists(ScenePath)) { EditorSceneManager.OpenScene(ScenePath); return; }
        EditorSceneManager.OpenScene("Assets/Scenes/MediaPipeTracking.unity");
        foreach (var component in UnityEngine.Object.FindObjectsByType<MonoBehaviour>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (component is CS_MediaPipeReceiver || component is CS_RandomMotion || component is DataCollectorBase ||
                component is CS_Operation || component is CommUSyudou) component.enabled = false;
        var target = UnityEngine.Object.FindAnyObjectByType<CS_TestRotateBones>(FindObjectsInactive.Include);
        var camera = Camera.main ?? UnityEngine.Object.FindAnyObjectByType<Camera>(FindObjectsInactive.Include);
        if (target == null || camera == null) throw new Exception("Source scene has no robot or camera");
        Activate(target.transform); Activate(camera.transform);
        target.enabled = true; camera.enabled = true;
        if (camera.GetComponent<AudioListener>() == null) camera.gameObject.AddComponent<AudioListener>();
        var comparison = new GameObject("VideoPoseComparison").AddComponent<VideoPoseComparison>();
        comparison.runFolder = DefaultRun; comparison.target = target; comparison.robotCamera = camera;
        EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), ScenePath);
        AssetDatabase.SaveAssets();
        Debug.Log("Created three-panel comparison scene: " + ScenePath);
    }

    static void Activate(Transform item)
    { for (; item != null; item = item.parent) item.gameObject.SetActive(true); }

    public static void RunBatch()
    {
        try
        {
            Open();
            var comparison = UnityEngine.Object.FindAnyObjectByType<VideoPoseComparison>();
            var run = new ComparisonRun(Argument("--comparison-run", comparison.runFolder));
            if (comparison.target == null || comparison.robotCamera == null) throw new Exception("Missing scene references");
            if (comparison.target.Bone.Skip(1).Any(x => x == null)) throw new Exception("Missing robot bones");
            var value = new float[14];
            run.Poses.Sample(run.Poses.Times[20], value);
            for (int j = 0; j < 14; j++) if (Mathf.Abs(value[j] - run.Poses.Angles[20][j]) > .001f) throw new Exception("CSV sample alignment");
            double middle = (run.Poses.Times[10] + run.Poses.Times[11]) / 2;
            run.Poses.Sample(middle, value);
            for (int j = 0; j < 14; j++) if (Mathf.Abs(value[j] - (run.Poses.Angles[10][j] + run.Poses.Angles[11][j]) / 2) > .001f) throw new Exception("CSV interpolation");
            Debug.Log($"Video comparison validation PASSED: scene, source files, joint order, {run.Poses.Count} poses, timestamps and interpolation.");
            EditorApplication.Exit(0);
        }
        catch (Exception error) { Debug.LogException(error); EditorApplication.Exit(1); }
    }

    static string Argument(string key, string fallback)
    {
        string[] args = Environment.GetCommandLineArgs();
        int index = Array.IndexOf(args, key);
        return index >= 0 && index + 1 < args.Length ? args[index + 1] : fallback;
    }

    static VideoPoseComparison renderer;
    static ComparisonRun renderRun;
    static RenderTexture renderTexture;
    static Texture2D pixels;
    static string output;
    static int frame, frameCount, warmup;
    static readonly float[] pose = new float[14];

    // Requires a graphics device: omit Unity's -nographics flag.
    public static void RenderBatch()
    {
        try
        {
            Open();
            renderer = UnityEngine.Object.FindAnyObjectByType<VideoPoseComparison>();
            renderRun = new ComparisonRun(Argument("--comparison-run", renderer.runFolder));
            output = Path.GetFullPath(Argument("--comparison-output", Path.Combine(ComparisonRun.RepositoryRoot,
                "outputs/video_comparison/00271_6f35a901156dd4cd1fdf/unity_frames")));
            Directory.CreateDirectory(output);
            int total = (int)Math.Ceiling(renderRun.Poses.Duration * 30 - 1e-6);
            frameCount = Math.Min(total, int.Parse(Argument("--comparison-frames", total.ToString())));
            if (frameCount < 1) throw new Exception("Render frame count must be positive");
            if (!renderer.target.BeginExternalControl()) throw new Exception("Cannot acquire robot for rendering");
            var robot = renderer.target.GetComponentInParent<CS_CommU>();
            if (robot != null) robot.enableLookAt = false;
            renderTexture = new RenderTexture(1280, 720, 24, RenderTextureFormat.ARGB32);
            renderTexture.Create();
            pixels = new Texture2D(1280, 720, TextureFormat.RGB24, false);
            frame = 0; warmup = 5;
            EditorApplication.update += RenderTick;
        }
        catch (Exception error) { Debug.LogException(error); FinishRender(1); }
    }

    static void RenderTick()
    {
        try
        {
            if (warmup-- > 0) { EditorApplication.QueuePlayerLoopUpdate(); return; }
            renderRun.Poses.Sample(frame / 30.0, pose);
            renderer.target.ApplyExternalPose(pose);
            RenderPipeline.SubmitRenderRequest(renderer.robotCamera,
                new UniversalRenderPipeline.SingleCameraRequest { destination = renderTexture });
            RenderTexture.active = renderTexture;
            pixels.ReadPixels(new Rect(0, 0, 1280, 720), 0, 0);
            pixels.Apply(); RenderTexture.active = null;
            File.WriteAllBytes(Path.Combine(output, frame.ToString("D6") + ".png"), pixels.EncodeToPNG());
            frame++;
            if (frame % 30 == 0) Debug.Log($"Rendered Unity comparison frame {frame}/{frameCount}");
            if (frame >= frameCount)
            {
                File.WriteAllText(Path.Combine(output, "render.json"), JsonUtility.ToJson(new RenderInfo {
                    frame_count = frameCount, fps = 30, csv = renderRun.Csv,
                    csv_sha256 = BitConverter.ToString(System.Security.Cryptography.SHA256.Create().ComputeHash(File.ReadAllBytes(renderRun.Csv))).Replace("-", "").ToLowerInvariant(), source = renderRun.Original,
                    generated = renderRun.Generated, source_offset_seconds = renderRun.OriginalOffset }, true));
                Debug.Log("Unity comparison rendering PASSED: " + output);
                FinishRender(0);
            }
        }
        catch (Exception error) { Debug.LogException(error); FinishRender(1); }
    }

    [Serializable] class RenderInfo
    { public int frame_count, fps; public string csv, csv_sha256, source, generated; public double source_offset_seconds; }

    static void FinishRender(int status)
    {
        EditorApplication.update -= RenderTick;
        if (renderer != null && renderer.target != null) renderer.target.EndExternalControl();
        if (renderTexture != null) { renderTexture.Release(); UnityEngine.Object.DestroyImmediate(renderTexture); }
        if (pixels != null) UnityEngine.Object.DestroyImmediate(pixels);
        EditorApplication.Exit(status);
    }
}
