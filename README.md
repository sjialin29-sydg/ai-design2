# ai-design2
brief
Training GPT2 Chinese from zero to hero
==

1.Description:
---
从头训练一个82M的中文GPT2模型，使用BERT的Tokenizer.中文语料采用斗破苍穹小说的部分章节，大小约16M。训练15个周期，batchsize=8。最终可以续写10句以上的斗破苍穹小说。

2.Start:
----
(1)***environment***

首先，我们下载依赖。
```bash
pip install -r requirements.txt
```

(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料由.txt文件更改为input.json文件
斗破苍穹小说语料来源于https://github.com/GaoPeng97/transformer-xl-chinese/tree/master/data/doupo

按照参考样例./train.json更改input.json文件格式,由于数据集内容为原始的小说内容，包含着大量的非法字符和json读取不支持的控制字符，因此我们对原始数据集文件进行处理，去除其中非法字符，生成预处理好的数据集文件train.json。
```bash
python clr_ctrl.py
```

(3)***Model***

在model_config 定义初始GPT-2模型的超参数配置，
- "initializer_range": 0.02 ： 定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为0，标准差为0.02的正态分布中进行随机初始化。
- "layer_norm_epsilon": 1e-05 ： 用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为1e-05，用于稳定训练。
- "n_ctx": 1024 ： 表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为1024，即模型一次最多能处理1024个 token。
- "n_embd": 768 ： 表示每个token的嵌入维度大小，即模型中词向量的维度。设置为768，即每个词汇的表示向量是768维的。
- "n_head": 12 ： 表示自注意力机制中的注意力头的数量。设置为12，即模型的多头注意力机制中有12个独立的头。
- "n_layer": 10 ： 表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
- "n_positions": 1024 ： 表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 n_ctx一致，表示模型最多能处理1024个位置的token。
- "vocab_size": 13317 ： 表示词汇表的大小，即模型可以识别和生成的词汇数量。在这里，词汇表大小为 21128，表示该模型可以处理的词汇量为21128个不同的 token。


(4)***Training***

现在，我们可以使用我们处理好的数据集来训练我们的初始gpt2模型，使用如下命令：
```bash
python train.py   --model_config config/model_config_small.json   --tokenized_data_path data/tokenized/   --tokenizer_path cache/vocab_small.txt   --raw_data_path data/train.json   --epochs 15   --log_step 200   --stride 512   --output_dir model/   --device 0,1   --num_pieces 100   --raw
```

在这个过程中，我们可以看到命令窗口打印出模型的config文件，定义了模型的结构；同时也打印出了模型的参数量，为81894144，约82M

Print Model config
config:
{
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 10,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 13317
}
number of parameters: 81894144

训练过程中，每个epoch对应的模型都将存储在./model/目录下，最终训练好的模型将存储在./model/final_model/路径中。

(5)***Generate***

现在，我们可以使用我们用目标语料训练生成的模型来进行文字生成，使用如下命令：
```bash
python generate.py   --device 1   --length 1000   --tokenizer_path cache/vocab_small.txt   --model_path model/final_model   --prefix "[CLS]萧炎大喝一声"   --topp 1   --temperature 1.0 --save_samples --save_samples_path ./mnt/
```

3.Result
--
最终会生成10个文字样本，存储在./mnt/目录下，其中之一如下：

======================================== SAMPLE 1 ========================================

萧炎大喝一声,背后青火双翼一振，便是猛的对着下方大部落,暴射而去!“嗤!”在青火急速消失的那一刻,萧炎身形陡然停顿,一头对着前方暴掠而去，与此同时，青鳞也是如同鬼魅般的出现在前者前方,阴狠的掌风,狠狠的拍出,最后重重的落在后者身体之上,当下,两人便是惊骇的发现,那股恐怖劲气,居然是将萧炎拍飞而去。“啊!”强悍劲风扩散而开,萧炎的身体，也是在半空中蹬蹬的连退,最后砸在一座巨石之上,强悍的劲力,几乎将地面震得粉碎，而萧炎的身体,则是诡异消失。“好恐怖的速度!”躲避开地面,萧炎眼角也是掠过一抹凝重,这天妖傀”不愧是远古天啸的存在，这等奇异能力,这种奇宝”若是换作寻常人，就算是拥有着灵智，都是无法察觉,若是放在天妖傀身体上的话,也会在瞬间被炸成粉末,这种程度”可真是太过惊险了。“嘭!”在萧炎心念转动间”天空上，一道金光陡然从天际暴掠而来,旋即狠狠的砸在其身体之上，可怕的劲力扩散而来。“噗嗤”金光撞击的那一霎,萧炎所化的金光身体，却是猛的倒射而出，最后重重的落在广场之上，几颗丈许粗的裂缝,也是在此刻爆裂而开。“好恐怖的破坏力不过这天妖傀是斗宗巅峰的实力,居然连妖瞑这等级的强者都不具”见到萧炎受创，青鳞也是忍不住的轻吸了一口凉气，这家伙的肉体强悍程度,远远超过了萧炎，想要达到这里，却是需要一些困难的事情。一拳头拍了拍萧炎的肩膀,萧炎眼中掠过一抹欣喜之意,这家伙,果然是有些难缠,只要一想到它都是能够彻底恢复,那样的话,这天妖傀，就算是一位半圣强者,恐怕都是得当场重伤。“萧炎兄弟,接下来,便让我们先行离去”古青阳大手拍了拍萧炎手臂上的小萧潇,笑道。闻言,萧炎也是忍不住的一笑,旋即点点头,身形一闪，便走出现在青阳几人面前,袖袍挥动,浩瀚的斗气便是如同潮水一般,自其体内暴涌而出,最后化为几道百丈庞大的斗气匹练,匹练闪电般的凝聚在萧炎二人面前。“好雄浑的斗气,这才两个小时间不够,也好满足了吧？”见到一下子便是陨落之巅,众人都是忍不住的倒吸了一口凉气,这天墓之中,果然是有些底蕴”，萧炎微微一笑,屈指一弹”一缕乳白色的火焰便是迎风暴涨,然后重重的轰在那道斗气匹练之上。“噗”接收的火柱，萧炎这才抬起头,目光望着那几乎呈燎原之势暴掠而来的斗圣强者,此刻的后者,明显已是处于了极度坚硬的地面之上，甚至,一些斗尊巅峰的强者,都足以和一名二星斗尊相媲美。“砰”萧炎手掌微握”强猛的力道,直接是将一名斗尊震得吐血倒飞而

==========================================================================================
