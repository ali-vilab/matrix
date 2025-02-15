# The Matrix (Preview)
<div align="center">
<img src=readme_src/white-logo.svg width="50%"/>
</div>
<p align="center">
Download The Matrix model weights at <a href="https://huggingface.co/MatrixTeam/TheMatrix" target="_blank"> 🤗 Huggingface</a> or <a href="https://www.modelscope.cn/models/AiurRuili/TheMatrix" target="_blank"> 🤖 ModelScope</a>
</p>
<p align="center">
📚 View the <a href="https://arxiv.org/abs/2412.03568" target="_blank">Paper</a>, <a href="https://matrixteam-ai.github.io/pages/TheMatrix/" target="_blank"> Website, and <a href="http://matrixteam-ai.github.io/docs/TheMatrixDocs" target="_blank">Documentation</a>
</p>
<p align="center">
    👋 Say Hi to our team and members at <a href="https://matrixteam-ai.github.io/" target="_blank">Matrix-Team</a>
</p>
<p align="center">
📍 (Coming Soon) Explore The Matrix playground online at <a href="">Journee</a> to experience real-time AI generated world.
</p>

<div style="text-align:center;"> 
    <video controls
           width="720"
           poster="[readme_src/font_30s.png](https://github.com/user-attachments/assets/51b8b2b0-be78-41c3-b4d3-5a2a66b59a0e)">
        <source src="https://github.com/user-attachments/assets/c695160a-48bc-43e3-8ffc-700271ac0084">
        Your browser does not support the video tag.
    </video>
</div>

---
## What is The Matrix?
The Matrix is an advanced world model designed to generate **high-quality, infinite-time interactive videos in real-time**, setting a new benchmark in the field of **neural interactive simulations**. It is simultaneously several innovations:

- **A cutting-edge world model** that generates continuous, interactive video content with **unparalleled realism and length**.
- A **real-time** system that supports infinite content generation, overcoming previous limitations seen in simpler 2D game models like DOOM or Minecraft.
- A powerful model architecture powered by the **Swin-DPM** model, designed to produce dynamic, **ever-expanding** content.
- A novel training strategy that integrates both real and simulated data, enhancing the system's ability to exceptional **generalization** capabilities.

At its core, The Matrix combines these elements to push the boundaries of interactive video generation, making real-time, high-quality, infinite-length content a reality.

## Documentation
Comprehensive documentation is available in [English](). This includes detailed installation steps, tutorials, and training instructions. The [paper](https://arxiv.org/abs/2412.03568) and [Project Page](https://matrixteam-ai.github.io/pages/TheMatrix/) offer more details about the method.


## Model Weights
Model checkpoints can be found in [Huggingface](https://huggingface.co/MatrixTeam/TheMatrix) and [ModelScope](https://www.modelscope.cn/models/AiurRuili/TheMatrix). Please refer to the [Documentation](http://matrixteam-ai.github.io/docs/TheMatrixDocs) for how to load them for inferences.


## Important Updates

According to a request from Alibaba Tongyi Lab, the previous version of The Matrix was inherited from an internal version of Video DiT and could not be openly released. Therefore, we have re-implemented The Matrix code based on the previously open-released video generation model, [CogVideoX](https://github.com/THUDM/CogVideo/tree/main). We sincerely appreciate the efforts of the **CogVideo** team for their contributions.

As a result, the open release of our model has been delayed, and some components are still under development. These components will be released as soon as they are finished, including:

- [ ] Inference scripts for 8-GPU parallel inference of the DiT backbone, which will accelerate the inference speed by around 6-8 times.

- [ ] Training of the Stream Consistency Models, which will accelerate inference speed by around 7-10 times.

- [ ] Training on fused realistic and simulated data to acquire stronger generalization ability.

## Reimplementation contributions
The successful release of The Matrix Project is built upon the collective efforts of our incredibly talented team members. We extend our heartfelt gratitude for their dedication, hard work, and invaluable contributions. Those members are:

**Longxiang Tang, Zhicai Wang, Ruili Feng, Ruihang Chu, Han Zhang, and Zhantao Yang**

Special Thanks to **Longxiang** and **Zhicai** for their excellent contributions.

## Additional Notes

There have been certain changes to the hyperparameter settings and training strategy compared to what is reported in the paper due to the re-implementation. Please be aware of these when reviewing the code.

Despite these changes, we are pleased to announce that the overall generation quality is much more advanced compared to the previous version after more careful design of methods and parameters.

## Citation
If you find our work useful please consider citing:

```bibtex
@article{feng2024matrix,
  title={The matrix: Infinite-horizon world generation with real-time moving control},
  author={Feng, Ruili and Zhang, Han and Yang, Zhantao and Xiao, Jie and Shu, Zhilei and Liu, Zhiheng and Zheng, Andy and Huang, Yukun and Liu, Yu and Zhang, Hongyang},
  journal={arXiv preprint arXiv:2412.03568},
  year={2024}
}
```

## License
The code in this repository is released under the Apache 2.0 License.

The Matrix model (including its corresponding Transformers module and VAE module) is released under the Apache 2.0 License.
