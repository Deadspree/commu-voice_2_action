using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using UnityEngine;
using System;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using UnityEngine.Networking;

public class CS_Operation : MonoBehaviour
{
    float[] sleep_list = {};
    float[] velocity_list = {};
    int[] index_list = {};
    float[] goal_list = {};
    public bool Operation_Flag = false;
    static System.Threading.SemaphoreSlim CommU_Lock = new System.Threading.SemaphoreSlim(1, 1);
    public string input_str;
    public static CS_Operation instance;

    public void Awake()
    {
        if (instance == null)
        {
            instance = this;
        }
    }

    void Start()
    {
        // StartCoroutine(AutoReadGesture());
    }

    void Update()
    {
        if (!string.IsNullOrEmpty(input_str))
        {
            StartCoroutine(ReadGestureFromFile(input_str));
            input_str = "";
        }
    }

    // private IEnumerator AutoReadGesture()
    // {
    //     while (true)
    //     {
    //         yield return new WaitForSeconds(UnityEngine.Random.Range(1f, 5f));
    //         Operation_Flag = true;
    //         StartCoroutine(ReadGestureFromFile("eyeblink"));
    //     }
    // }
    // private bool IsAutoReading = false;

    // private IEnumerator AutoReadGesture()
    // {
    //     if (IsAutoReading) yield break; // すでに実行中ならスキップ
    //     IsAutoReading = true;

    //     while (true)
    //     {
    //         yield return new WaitForSeconds(UnityEngine.Random.Range(1f, 5f));

    //         if (!Operation_Flag) // すでに動作中ならスキップ
    //         {
    //             Operation_Flag = true;
    //             StartCoroutine(ReadGestureFromFile("eyeblink"));
    //         }
    //     }
    // }

    private IEnumerator ReadGestureFromFile(string fileName)
    {
        string path = Path.Combine(Application.streamingAssetsPath, "gesture", fileName + ".s3r");
        // string path = Application.streamingAssetsPath + "/gesture/" + fileName + ".s3r";
    
        #if UNITY_EDITOR || UNITY_STANDALONE
            // Editor や PC向けは file:// を付ける
            string url = "file://" + path;
        #else
            // Android / iOS は streamingAssetsPath が特殊なのでそのまま
            string url = path;
        #endif

        Debug.Log("Checking for file: " + url);

        UnityWebRequest request = UnityWebRequest.Get(url);
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string data = request.downloadHandler.text;
            Debug.Log("Read Gesture Data: " + data);
            ReadGesture(data);
        }
        else
        {
            Debug.LogWarning("Gesture file not found: " + url + " Error: " + request.error);
        }
    }

    private void ReadGesture(string input)
    {
        string[] input_line = input.Split('\n');
        Debug.Log("Processing Gesture: " + input_line[0]);

        foreach (string line in input_line)
        {
            if (!string.IsNullOrEmpty(line))
            {
                AddGesture(line);
            }
        }
    }

    private void AddGesture(string input)
    {
        string[] input_list = Regex.Split(input, "\t");
        switch (input_list[1])
        {
            case "P":
                Debug.Log("P");
                int move_num = (input_list.Length - 5) / 2;
                for (int i = 0; i < move_num; i++)
                    if (i > 0)
                        AddList(0f, float.Parse(input_list[3]), int.Parse(input_list[4 + 2 * i]), float.Parse(input_list[5 + 2 * i]));
                    else
                        AddList(float.Parse(input_list[0]), float.Parse(input_list[3]), int.Parse(input_list[4 + 2 * i]), float.Parse(input_list[5 + 2 * i]));
                break;
            case "t":
                Debug.Log("t");
                AwakeGesture();
                // await Stop(float.Parse(input_list[0]));
                StartCoroutine(StopCoroutine(float.Parse(input_list[0])));

                break;
        }
    }

    void AwakeGesture()
    {
        CS_TestRotateBones.instance.Gesture(sleep_list, velocity_list, index_list, goal_list);
        ClearList();
    }

    void AddList(float sleep, float velocity, int index, float goal)
    {
        Array.Resize(ref sleep_list, sleep_list.Length + 1);
        Array.Resize(ref velocity_list, velocity_list.Length + 1);
        Array.Resize(ref index_list, index_list.Length + 1);
        Array.Resize(ref goal_list, goal_list.Length + 1);

        sleep_list[sleep_list.Length - 1] = sleep;
        velocity_list[velocity_list.Length - 1] = velocity;
        index_list[index_list.Length - 1] = index;
        goal_list[goal_list.Length - 1] = goal;
    }

    // private async Task Stop(float sleep)
    // {
    //     await Task.Delay((int)(sleep * 1000));
    // }
    private IEnumerator StopCoroutine(float sleep)
    {
        yield return new WaitForSeconds(sleep);
    }

    void ClearList()
    {
        Array.Resize(ref sleep_list, 0);
        Array.Resize(ref velocity_list, 0);
        Array.Resize(ref index_list, 0);
        Array.Resize(ref goal_list, 0);
    }
}



// ビルド後にはローカルファイルにアクセスできない

// using System.Collections;
// using System.Collections.Generic;
// using System.Text.RegularExpressions;
// using UnityEngine;
// using System;
// using System.Text;
// using System.Threading.Tasks;
// using System.IO;

// public class CS_Operation : MonoBehaviour
// {
//     float[] sleep_list = {};
//     float[] velocity_list = {};
//     int[] index_list = {};
//     float[] goal_list = {};
//     public bool Operation_Flag = false;
//     static System.Threading.SemaphoreSlim CommU_Lock = new System.Threading.SemaphoreSlim(1, 1);
//     public string input_str;
//     public static CS_Operation instance;

//     // public string gestureFileName = "lipAsynchOnce.s3r"; // 読み込むジェスチャーファイル

//     public void Awake()
//     {
//         if (instance == null)
//         {
//             instance = this;
//         }
//     }

//     void Start()
//     {
//         StartCoroutine(AutoReadGesture()); // ランダムな間隔でReadGestureを実行
//     }

//     void Update()
//     {
//         if(!string.IsNullOrEmpty(input_str))
//         {
//             string path = $"./Assets/gesture/{input_str}.s3r";
//             Debug.Log("Checking for file: " + path);

//             if (File.Exists(path))
//             {
//                 string data = File.ReadAllText(path);
//                 Debug.Log("Read Gesture Data: " + data);
//                 ReadGesture(data);
//             }
//             else
//             {
//                 Debug.LogWarning("Gesture file not found: " + path);
//             }
//             input_str = "";
//         }
//     }
//     private IEnumerator AutoReadGesture()
//     {
//         while (true)
//         {
//             yield return new WaitForSeconds(UnityEngine.Random.Range(1f, 5f));

//             Operation_Flag = true;
//             // string path = $"./Assets/gesture/{gestureFileName}";
//             string path = $"./Assets/gesture/eyeblink.s3r";
//             Debug.Log("Checking for file: " + path);

//             if (File.Exists(path))
//             {
//                 string data = File.ReadAllText(path);
//                 Debug.Log("Read Gesture Data: " + data);
//                 ReadGesture(data);
//             }
//             else
//             {
//                 Debug.LogWarning("Gesture file not found: " + path);
//             }
//         }
//     }

//     private void ReadGesture(string input)
//     {
//         string[] input_line = input.Split('\n');
//         Debug.Log("Processing Gesture: " + input_line[0]);

//         for (int i = 0; i < input_line.Length; i++)
//         {
//             if (!string.IsNullOrEmpty(input_line[i]))
//             {
//                 AddGesture(input_line[i]);
//             }
//         }
//     }

//     private async void AddGesture(string input)
//     {
//         string[] input_list = Regex.Split(input, "\t");
//         switch (input_list[1])
//         {
//             case "P":
//                 Debug.Log("P");
//                 int move_num = (input_list.Length - 5) / 2;
//                 for (int i = 0; i < move_num; i++)
//                     if (i > 0)
//                         AddList(0f, float.Parse(input_list[3]), int.Parse(input_list[4 + 2 * i]), float.Parse(input_list[5 + 2 * i]));
//                     else
//                         AddList(float.Parse(input_list[0]), float.Parse(input_list[3]), int.Parse(input_list[4 + 2 * i]), float.Parse(input_list[5 + 2 * i]));
//                 break;
//             case "t":
//                 Debug.Log("t");
//                 AwakeGesture();
//                 await Stop(float.Parse(input_list[0]));
//                 break;
//         }
//     }

//     void AwakeGesture()
//     {
//         CS_TestRotateBones.instance.Gesture(sleep_list, velocity_list, index_list, goal_list);
//         ClearList();
//     }

//     void AddList(float sleep, float velocity, int index, float goal)
//     {
//         Array.Resize(ref sleep_list, sleep_list.Length + 1);
//         Array.Resize(ref velocity_list, velocity_list.Length + 1);
//         Array.Resize(ref index_list, index_list.Length + 1);
//         Array.Resize(ref goal_list, goal_list.Length + 1);

//         sleep_list[sleep_list.Length - 1] = sleep;
//         velocity_list[velocity_list.Length - 1] = velocity;
//         index_list[index_list.Length - 1] = index;
//         goal_list[goal_list.Length - 1] = goal;
//     }

//     private async Task Stop(float sleep)
//     {
//         await Task.Delay((int)(sleep * 1000));
//     }

//     void ClearList()
//     {
//         Array.Resize(ref sleep_list, 0);
//         Array.Resize(ref velocity_list, 0);
//         Array.Resize(ref index_list, 0);
//         Array.Resize(ref goal_list, 0);
//     }
// }




// 宮原さんのプログラム↓

// using System.Collections;
// using System.Collections.Generic;
// using System.Text.RegularExpressions;
// using UnityEngine;
// using System;
// using System.Text;
// using System.Threading.Tasks;
// using System.IO;

// public class CS_Operation : MonoBehaviour
// {

//     float[] sleep_list = {};
//     float[] velocity_list = {};
//     int[] index_list = {};
//     float[] goal_list = {};
//     public bool Operation_Flag = false;
//     static System.Threading.SemaphoreSlim CommU_Lock = new System.Threading.SemaphoreSlim(1, 1);
//     public string input_str;
//     public static CS_Operation instance;
//     // Start is called before the first frame update

//     public void Awake()
//     {
//         if (instance == null)
//         {
//             instance = this;
//         }
//     }

//     void Start()
//     {
        
//     }

//     // Update is called once per frame
//     async void Update()
//     {
//         /*
//         if (Input.GetKeyDown(KeyCode.Q)){
//             CS_TestRotateBones.instance.OnDownButton(0.0f, 60.0f, 6, 6.0f);
//         }

//         if (Input.GetKeyDown(KeyCode.W))
//         {
//             CS_TestRotateBones.instance.OnDownButton(0.5f, 60.0f, 6, -12.0f);
//         }

//         if (Input.GetKeyDown(KeyCode.E))
//         {
//             CS_TestRotateBones.instance.OnDownButton(0.5f, 60.0f, 6, 0.0f);
//         }

//         if (Input.GetKeyDown(KeyCode.Z))
//         {
//             CS_TestRotateBones.instance.OnDownButton(0.0f, 60.0f, 6, 6.0f);
//             CS_TestRotateBones.instance.OnDownButton(0.5f, 60.0f, 6, -12.0f);
//             CS_TestRotateBones.instance.OnDownButton(0.5f, 60.0f, 6, 0.0f);
//         }
//         */
        

//         if (Input.GetKeyDown(KeyCode.Q))
//         {
//             string input = "0.0	P	0.0	70	4	30	2	30	-1";
//             await CommU_Lock.WaitAsync();
//             try
//             {
//                 Debug.Log("try");
//                 ReadGesture(input);
//             }
//             finally
//             {
//                 CommU_Lock.Release();
//             }

//             /*
//             string input;
//             input = "0.0	P	0.0	70	4	30	2	30	-1";
//             AddGesture(input);
            
//             input = "2.0	P	0.0	20	5	-20	3	20	-1";
//             AddGesture(input);
//             input = "0.7	P	0.0	20	5	5	3	-5	-1";
//             AddGesture(input);
//             input = "0.7	P	0.0	20	5	-20	3	20	-1";
//             AddGesture(input);
//             input = "0.7	P	0.0	20	5	5	3	-5	-1";
//             AddGesture(input);
//             input = "0.7	P	0.0	70	4	-90	2	-90	-1";
//             AddGesture(input);
//             input = "0.1	t";
//             AddGesture(input);

//             //AwakeGesture();
//             */
            
//         }
//         if (Input.GetKeyDown(KeyCode.E))
//         {
//             string input = "0.0	P	0.0	70	4	0	2	0	-1\n0.1	t";
//             AddGesture(input);
//             AwakeGesture();
//         }

//         if (Input.GetKeyDown(KeyCode.T))
//         {
//             Operation_Flag = true;
//             string path = "./Assets/gesture/crap.s3r";
//             Debug.Log(path);
//             // �ǂݍ���
//             string data = File.ReadAllText(path);
//             Debug.Log("Data is " + data);
//             Debug.Log("try");
//             ReadGesture(data);
//         }
//         if (Input.GetKeyDown(KeyCode.A))
//         {
//             Operation_Flag = true;
//             string path = "./Assets/gesture/eyeblink.s3r";
//             Debug.Log(path);
//             // �ǂݍ���
//             string data = File.ReadAllText(path);
//             Debug.Log("Data is " + data);
//             Debug.Log("try");
//             ReadGesture(data);
//         }
//         if (Input.GetKeyDown(KeyCode.Y))
//         {
//             Operation_Flag = true;
//             string path = "./Assets/gesture/baibai.s3r";
//             Debug.Log(path);
//             // �ǂݍ���
//             string data = File.ReadAllText(path);
//             Debug.Log("Data is " + data);
//             Debug.Log("try");
//             ReadGesture(data);
//         }

//         if (Input.GetKeyDown(KeyCode.P))
//         {
//             Debug.Log(CS_TestRotateBones.instance.Bone[4].localEulerAngles);
//         }

//         if (File.Exists("./Assets/gesture/" + input_str + ".s3r") == true)
//         {
//             Operation_Flag = true;
//             string path = "./Assets/gesture/" + input_str + ".s3r";
//             Debug.Log(path);
//             // �ǂݍ���
//             string data = File.ReadAllText(path);
//             // Debug.Log("Data is " + data);
//             ReadGesture(data);
//             input_str = "";
//         }

//     }


//     private void ReadGesture(string input)
//     {
//         string[] del = { "\n" };

//         //string[] input_line = input.Split(del, StringSplitOptions.None);
//         string[] input_line = input.Split('\n');
//         Debug.Log(input_line[1]);
//         for (int i = 0; i < input_line.Length; i++)
//         {
//             Debug.Log(input_line[i]);
//             if(input_line[i]!="")
//                 AddGesture(input_line[i]);
//         }

//     }


//     void AwakeGesture()
//     {
//         /*
//         AddList(0.0f, 60.0f, 6, 6.0f);
//         AddList(0.5f, 60.0f, 6, -12.0f);
//         AddList(0.5f, 60.0f, 6, 0.0f);

//         float[] sleep_list = { 0.0f, 0.5f, 0.5f };
//         float[] velocity_list = { 60.0f, 60.0f, 60.0f };
//         int[] index_list = { 6, 6, 6 };
//         float[] goal_list = { 6.0f, -12.0f, 0.0f };
//         */
//         CS_TestRotateBones.instance.Gesture(sleep_list, velocity_list, index_list, goal_list);
//         ClearList();
//     }


//     private async void AddGesture(string input)
//     {
//         string[] input_list = Regex.Split(input, "\t");
//         switch (input_list[1])
//         {
//             case "P":
//                 Debug.Log("P");
//                 int move_num = (input_list.Length - 5) / 2;
//                 for (int i = 0; i < move_num; i++)
//                     if (i > 0)
//                         AddList(0f, float.Parse(input_list[3]), int.Parse(input_list[4 + 2 * i]), float.Parse(input_list[5 + 2 * i]));
//                     else
//                         AddList(float.Parse(input_list[0]), float.Parse(input_list[3]), int.Parse(input_list[4 + 2 * i]), float.Parse(input_list[5 + 2 * i]));
//                 break;
//             case "t":
//                 Debug.Log("t");
//                 AwakeGesture();
//                 await Stop(float.Parse(input_list[0]));
//                 //Operation_Flag = false;
//                 break;
//             default:
//                 break;
//         }

//     }

//     void AddList(float sleep, float velocity, int index, float goal)
//     {
//         Array.Resize(ref sleep_list, sleep_list.Length + 1);
//         Array.Resize(ref velocity_list, velocity_list.Length + 1);
//         Array.Resize(ref index_list, index_list.Length + 1);
//         Array.Resize(ref goal_list, goal_list.Length + 1);

//         sleep_list[sleep_list.Length - 1] = sleep;
//         velocity_list[velocity_list.Length - 1] = velocity;
//         index_list[index_list.Length - 1] = index;
//         goal_list[goal_list.Length - 1] = goal;
//     }

//     private async Task Stop(float sleep)
//     {
//         await Task.Delay((int)(sleep * 1000));
//     }

//     void ClearList()
//     {
//         Array.Resize(ref sleep_list, 0);
//         Array.Resize(ref velocity_list, 0);
//         Array.Resize(ref index_list, 0);
//         Array.Resize(ref goal_list, 0);
//     }

// }
