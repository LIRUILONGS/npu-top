# 发布到 PyPI

包名：`npu-top`。首次发布版本：`0.1.0`。

## 一次性配置

1. 登录 [PyPI](https://pypi.org/) 并开启两步验证。
2. 打开 [Publishing](https://pypi.org/manage/account/publishing/) 页面。
3. 添加一个 pending publisher，字段必须与仓库一致：

   | 字段 | 值 |
   |---|---|
   | PyPI Project Name | `npu-top` |
   | Owner | `LIRUILONGS` |
   | Repository name | `npu-top` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

4. GitHub 仓库的 `Settings -> Environments` 中创建 `pypi` 环境。可以配置
   required reviewers，发布时会等待人工批准。

Trusted Publishing 使用 OIDC，不需要把 PyPI API Token 写进仓库或 Secrets。

## 发布新版本

1. 修改 `pyproject.toml` 中的 `version`。PyPI 不允许覆盖已上传的版本号。
2. 本地验证：

   ```bash
   python -m py_compile npu_top.py
   python -m unittest discover -s tests -v
   python -m pip wheel --no-deps --no-build-isolation -w dist .
   ```

3. 提交并推送：

   ```bash
   git add pyproject.toml
   git commit -m "release: vX.Y.Z"
   git push origin main
   ```

4. 在 GitHub 创建 Release，tag 使用 `vX.Y.Z`，然后发布。
5. `.github/workflows/publish.yml` 会执行测试、构建、`twine check`，并通过
   Trusted Publishing 上传到 PyPI。

也可以在 Actions 页面手动运行 `publish` workflow。手动运行适合首次发布前
验证配置，但同样会真实上传，版本号不能复用。

## 发布后验证

```bash
python -m pip index versions npu-top
python -m pip install --no-cache-dir npu-top==X.Y.Z
npu-top --version
```

PyPI 页面：<https://pypi.org/project/npu-top/>
