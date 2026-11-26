# Third-party notices

The ConvNeXt V2-Atto backbone follows [Meta's ConvNeXt V2 implementation](https://github.com/facebookresearch/ConvNeXt-V2). The upstream code is MIT-licensed. The ImageNet checkpoints are distributed by Meta under CC BY-NC 4.0 and are referenced by download links. See the [upstream license](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/LICENSE).

## ConvNeXt V2 MIT notice

MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Runtime dependencies

- [Mamba (state-spaces/mamba)](https://github.com/state-spaces/mamba) supplies the optional SSD CUDA backend via its installed Python package.
- [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision) provide the training runtime and ResNet implementations.
