# ComfyUI_OmniParser
Try [OmniParser](https://github.com/microsoft/OmniParser) in ComfyUI which a simple screen parsing tool towards pure vision based GUI agent.

----

**Notice 2024/12/06**
* 因为这个方法调用了ultralytics库，所以如果你在2024/12/04-12/05更新了ultralytics或者安装了ultralytics，请务必检查安装的ultralytics版本是否是8.3.41版本，如果是，请及时删除。查看方法:pip
  show ultralytics
* Because this method calls the ‘ultralytics’ library, if you updated ‘ultralytic’s or installed ‘ultralytics‘ on December 5, 2024 or December 5, 2024  , please make sure to check if the installed version of 'ultralytics' is 8.3.41. If so, please delete it!!  Viewing method: pip show ultralytics  



1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_OmniParser.git
```  

----
  
2.requirements  
----

```
pip install -r requirements.txt

```

----

3.Checkpoints
----
[huggingface-OmniParser](https://huggingface.co/microsoft/OmniParser)

----

4.Example
----
 
 ![](https://github.com/smthemex/ComfyUI_OmniParser/blob/main/example.png)


----

5.Citation
------
microsoft/OmniParser
```
@misc{lu2024omniparserpurevisionbased,
      title={OmniParser for Pure Vision Based GUI Agent}, 
      author={Yadong Lu and Jianwei Yang and Yelong Shen and Ahmed Awadallah},
      year={2024},
      eprint={2408.00203},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00203}, 
}
```
Some codes form [#](https://github.com/microsoft/OmniParser/pull/53) @aliencaocao
