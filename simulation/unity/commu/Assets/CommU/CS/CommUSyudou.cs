using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CommUSyudou : MonoBehaviour
{
    public Button optionButton1;
    public Button optionButton2;

    void Start()
    {
        optionButton1.onClick.AddListener(OnClick);
        optionButton2.onClick.AddListener(OnClick);
    }
    void OnClick()
    {
        CS_Operation.instance.input_str = "right_arm_up";
        StartCoroutine(WaitAndExecuteNext());
        // if (Input.GetKeyDown(KeyCode.M))
        // {
        //     CS_Operation.instance.input_str = "ojigi";
        // }
        // if (Input.GetKeyDown(KeyCode.M))
        // {
        //     CS_Operation.instance.input_str = "baibai";
        // }
    }
    private IEnumerator WaitAndExecuteNext()
    {
        yield return new WaitForSeconds(3f);
        CS_Operation.instance.input_str = "default";
    }
}
