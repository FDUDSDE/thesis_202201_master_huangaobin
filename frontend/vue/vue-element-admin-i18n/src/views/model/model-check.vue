<template>

  <div class="app-container">
    
    <h3>{{title}}</h3>
    <el-table
    :data="list1"
    stripe
    style="width: 100%">
    <el-table-column
      prop="name"
      label="种子选择器"
      width="160">
      </el-table-column>
      <el-table-column
      prop="create_time"
      label="创建时间"
      width="200">
      </el-table-column>
      <el-table-column
      prop="train_epoch"
      label="训练轮次"
      width="150">
      </el-table-column>
      <el-table-column
      prop="batch_size"
      label="批数据大小"
      width="150">
      </el-table-column>
      <el-table-column
      prop="learning_rate"
      label="学习率"
      width="150">
      </el-table-column>
      <el-table-column
      prop="pairs"
      label="点对数"
      width="150">
      </el-table-column>
      <el-table-column
      prop="train_graph"
      label="训练图数据"
      width="180">
      </el-table-column>
      <el-table-column
      prop="train_comms"
      label="训练群组"
      width="180">
      </el-table-column>
      <el-table-column
      prop="train_size"
      label="训练集大小"
      width="150">
      </el-table-column>
  </el-table>

<el-table
    :data="list2"
    stripe
    style="width: 100%">
    <el-table-column
      prop="name"
      label="局部扩张器"
      width="160">
      </el-table-column>
      <el-table-column
      prop="create_time"
      label="创建时间"
      width="200">
      </el-table-column>
      <el-table-column
      prop="train_epoch"
      label="训练轮次"
      width="150">
      </el-table-column>
      <el-table-column
      prop="batch_size"
      label="批数据大小"
      width="150">
      </el-table-column>
      <el-table-column
      prop="learning_rate"
      label="学习率"
      width="150">
      </el-table-column>
      <el-table-column
      prop="ada_size"
      label="ADAsize"
      width="150">
      </el-table-column>
      <el-table-column
      prop="train_graph"
      label="训练图数据"
      width="180">
      </el-table-column>
      <el-table-column
      prop="train_comms"
      label="训练群组"
      width="180">
      </el-table-column>
      <el-table-column
      prop="train_size"
      label="训练集大小"
      width="150">
      </el-table-column>
  </el-table>

  </div>

</template>

<script>

import {mapState, mapMutations, mapActions} from 'vuex'
export default {
  computed: {
    ...mapState('model', ['title','list1','list2'])
  },
  
  created() {
    this.fetchModel()
  },
  methods: {
    ...mapActions('model', ['fetchModel']),
  }
}
</script>
