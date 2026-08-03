using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
using UnityEngine.InputSystem;
#endif

public class CS_RandomMotion : MonoBehaviour
{
    [Header("Target")]
    public CS_TestRotateBones target;
    public bool autoFindTarget = true;
    [Header("Data Collector")]
    public DataCollect dataCollector;
    public bool autoFindDataCollector = true;

    [Header("Randomization")]
    public float intervalSeconds = 1.0f;
    public int jointsPerEvent = 3;
    [Tooltip("If >0, stop after this many randomized poses have been generated (0 = run forever until toggled).")]
    public int maxRandomPoses = 0;
    public float minVelocity = 10f;
    public float maxVelocity = 80f;
    public bool startOnAwake = false;
    public KeyCode toggleKey = KeyCode.R;

    Coroutine _routine;
    int _posesSent = 0;

    void Awake()
    {
        if (target == null && CS_TestRotateBones.instance != null)
            target = CS_TestRotateBones.instance;
        Debug.Log($"CS_RandomMotion.Awake target={(target!=null?target.name:"null")}");
    }

    void Start()
    {
        Debug.Log($"CS_RandomMotion.Start startOnAwake={startOnAwake}");
        if (target == null && autoFindTarget && CS_TestRotateBones.instance != null)
            target = CS_TestRotateBones.instance;

        if (target == null && autoFindTarget)
            target = FindObjectOfType<CS_TestRotateBones>();

        if (dataCollector == null && autoFindDataCollector)
            dataCollector = FindObjectOfType<DataCollect>();

        if (dataCollector == null && autoFindDataCollector)
            Debug.LogWarning("CS_RandomMotion.Start: DataCollect not found; set DataCollect in Inspector if you want automatic captures.");

        if (target == null)
            Debug.LogWarning("CS_RandomMotion.Start: target still null — assign in Inspector or ensure CS_TestRotateBones exists in scene.");

        if (startOnAwake)
            StartRandom();
    }

    void Update()
    {
        bool togglePressed = false;

#if ENABLE_INPUT_SYSTEM && !ENABLE_LEGACY_INPUT_MANAGER
        if (Keyboard.current != null)
        {
            Key parsedKey;
            if (System.Enum.TryParse<Key>(toggleKey.ToString(), out parsedKey))
            {
                var control = Keyboard.current[parsedKey];
                if (control != null && control.wasPressedThisFrame)
                    togglePressed = true;
            }
        }
#endif

#if !ENABLE_INPUT_SYSTEM || ENABLE_LEGACY_INPUT_MANAGER
        if (Input.GetKeyDown(toggleKey))
            togglePressed = true;
#endif

        if (togglePressed)
        {
            if (_routine == null) StartRandom();
            else StopRandom();
        }
    }

    public void StartRandom()
    {
        if (_routine == null)
        {
            Debug.Log("CS_RandomMotion: Starting random loop");
            _posesSent = 0;
            if (dataCollector != null)
            {
                dataCollector.PauseAutoCapture();
                Debug.Log("CS_RandomMotion: paused dataCollector auto-capture");
            }
            _routine = StartCoroutine(RandomLoop());
        }
    }

    public void StopRandom()
    {
        if (_routine != null)
        {
            StopCoroutine(_routine);
            _routine = null;
            if (dataCollector != null)
            {
                dataCollector.ResumeAutoCapture();
                Debug.Log("CS_RandomMotion: resumed dataCollector auto-capture");
            }
        }
    }

    IEnumerator RandomLoop()
    {
        int jointCount = CS_JointLimitDefinition.DefaultLimits.Length;

        while (true)
        {
            if (target == null && autoFindTarget)
                target = CS_TestRotateBones.instance != null ? CS_TestRotateBones.instance : FindObjectOfType<CS_TestRotateBones>();

            if (target != null)
            {
                int k = Mathf.Clamp(jointsPerEvent, 1, jointCount);
                var task = SendRandomGestureOnce(jointCount);
                // wait until the gesture Task completes and actual motion finishes
                while ((task != null && !task.IsCompleted) || (target != null && target.IsMoving()))
                    yield return null;

                // capture during a 1s stop
                if (dataCollector != null)
                {
                    Debug.Log("CS_RandomMotion: invoking dataCollector.CaptureSample()");
                    dataCollector.CaptureSample();
                }
                else
                {
                    Debug.LogWarning("CS_RandomMotion: no dataCollector assigned; skipping capture.");
                }

                // increment generated pose count and stop if we've reached the limit
                _posesSent++;
                if (maxRandomPoses > 0 && _posesSent >= maxRandomPoses)
                {
                    Debug.Log($"CS_RandomMotion: reached maxRandomPoses={maxRandomPoses}, stopping random loop");
                    if (dataCollector != null)
                    {
                        dataCollector.ResumeAutoCapture();
                        Debug.Log("CS_RandomMotion: resumed dataCollector auto-capture");
                    }
                    _routine = null;
                    yield break;
                }

                yield return new WaitForSeconds(1.0f);
            }
            else
            {
                Debug.LogWarning("CS_RandomMotion: target is null, not sending gestures");
            }

            // loop immediately to send next gesture once previous completes
            yield return null;
        }
    }

    // Send one random gesture (factored out so we can trigger manually)
    public void TriggerOnce()
    {
        int jointCount = CS_JointLimitDefinition.DefaultLimits.Length;
        if (target == null)
        {
            Debug.LogWarning("CS_RandomMotion.TriggerOnce: target is null");
            return;
        }
        StartCoroutine(TriggerOnceRoutine(jointCount));
    }

    IEnumerator TriggerOnceRoutine(int jointCount)
    {
        if (dataCollector != null)
        {
            dataCollector.PauseAutoCapture();
        }

        var task = SendRandomGestureOnce(jointCount);
        while ((task != null && !task.IsCompleted) || (target != null && target.IsMoving()))
            yield return null;

        if (dataCollector != null)
        {
            // capture during 1s stop
            dataCollector.CaptureSample();
            yield return new WaitForSeconds(1.0f);
            dataCollector.ResumeAutoCapture();
        }
    }

    System.Threading.Tasks.Task SendRandomGestureOnce(int jointCount)
    {
        int k = Mathf.Clamp(jointsPerEvent, 1, jointCount);
        float[] sleep_list = new float[k];
        float[] velocity_list = new float[k];
        int[] index_list = new int[k];
        float[] goal_list = new float[k];

        var chosen = new HashSet<int>();
        for (int i = 0; i < k; i++)
        {
            int idx;
            int tries = 0;
            do
            {
                idx = Random.Range(0, jointCount);
                tries++;
            } while (chosen.Contains(idx) && tries < 10);

            chosen.Add(idx);
            index_list[i] = idx;
            sleep_list[i] = 0f;
            velocity_list[i] = Random.Range(minVelocity, maxVelocity);

            var limit = CS_JointLimitDefinition.DefaultLimits[idx];
            goal_list[i] = Random.Range(limit.Min, limit.Max);
        }

        Debug.Log($"CS_RandomMotion: sending Gesture k={k} indices=[{string.Join(",", index_list)}] goals=[{string.Join(",", goal_list)}]");
        return target.Gesture(sleep_list, velocity_list, index_list, goal_list);
    }
}
