# ai-design2
brief
Training GPT2 Chinese from zero to hero
==

1.Description:
---
从头训练一个82M的中文GPT2模型，使用BERT的Tokenizer.中文语料采用凡人修仙传小说的部分章节，大小约32.5M。训练15个周期，batchsize=8。最终可以续写10句以上的斗破苍穹小说。

2.Start:
----
(1)***environment***

首先，我们下载依赖。
```bash
pip install -r requirements.txt
```

(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料由.txt文件更改为input.json文件
凡人修仙传小说语料来源于zlibrary

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

======================================== SAMPLE ========================================
韩立转身一看，却面无表情的回道。“不可能！若不是我二人联手，说不定还真要和他们争斗上一斗的。但是现在，我二人联手下倒不是什么难做的。”银月嫣然一笑的说道。“这也简单。我们也不用多想什么，只是将一些外来修士的神通给几人看看的。”韩立轻笑一声，随手一道法诀打在了银月体表之上，同时口中吩咐了一声。“是，主人！”银月娇躯答应一声。韩立点点头，身躯一个模糊，就再次化为一团银光的飞射而出，几个闪动后，就到了银月身前处，才蓦然单手一拍自己天灵盖。银月身上霞光闪动，一层层的五色灵雾一冒而出，将银月护在了其中。韩立也不说话，身上一阵五色灵光闪过后，竟化为了一名身穿五色羽衣的貌美女子。此女身材普通，但容颜秀丽，一身雪白如玉，单手持着一件八卦镜和一件檀香的绝色天衣祭了出来，一副绝不会真将银月二人彻底罩在其下的模样。而就在这些宝物狂注之下，韩立幻化人形的身影出现在银月身前，一对银白玉足下意识的扫了一眼面前的银月，玉容上现出一丝轻笑，但目光闪动不已。“我是谁，是慕兰人追杀你的那位慕兰人的奸细！”“慕兰人？”韩立一眼看出了银月的真容，不禁有些意外。银月一怔，尚未明白怎么回事时，娇声音就先传了过来：“韩兄，慕兰人已经到了，我们过去看看再说吧。”“你知道就好。”韩立一怔，马上厉声喝道。第八百四十九章追之“主人，去了！你也知道的，就在那边静静等你了。”银月笑容一敛，有些不太好意思的吩咐道。“是，主人！”韩立没有迟疑什么，当即一催法诀，周身青光大放，就化为一道青虹，破空而去。慕沛灵也不多言，同样化为一道惊虹，紧跟了过去。韩立站在原地，望着前方远去的遁光，脸上闪过一丝异色。“咦！那里竟有人，看来主人已经在那里了。”银月一见韩立遁光远去，有些意外起来。韩立心中一动，正想再问些什么时，远处的银月却抢先开口了。“小婢还在远处等了，那些人已经离开了这里。”韩立看了银月一眼，缓缓说道。“你真的看到了，这里的人迹全无。不过这里距离有点远了。以前辈的神通，真的有些不太好辨认，我也无需冒险。”银月轻笑一声。“小婢也就算了。我现在是银月，也不想和主人在这里耗费太多时时间。”韩立淡淡的说道。“小婢自然知道，你也在这里了。主人多留几日，我就离开这里了。”银月抿嘴一笑，娇嗔的说道。“这个自然！”韩立心中一动，神念急忙在此女身上一扫。白狐则不动声色地跟了上去。这时银月已经跟在韩立身后，一副不置可否的样子，不知她用了什么方法，竟真将韩立从未发
==========================================================================================
