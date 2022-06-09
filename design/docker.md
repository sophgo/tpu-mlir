# 如何更新docker

直接用docker/build.sh生成docker时间太久，不推荐。
推荐步骤如下：

1. 在Dockerfile_update中，FROM最新image，再填入需要更新的内容

2. 在update.sh中增加image版本号

3. 执行update.sh

4. docker push 最新image到dockerhub
