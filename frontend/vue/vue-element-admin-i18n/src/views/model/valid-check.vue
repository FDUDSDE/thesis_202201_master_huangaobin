<template>

  <div class="app-container">
    
    <h3>{{titlevalid}}</h3>
    <el-table
    :data="valid_results"
    stripe
    style="width: 100%">
    <el-table-column
      prop="selector"
      label="种子选择器"
      width="160">
      </el-table-column>
      <el-table-column
      prop="expender"
      label="局部扩张器"
      width="160">
      </el-table-column>
      <el-table-column
      prop="output_nums"
      label="输出个数"
      width="150">
      </el-table-column>
      <el-table-column
      prop="valid_graph"
      label="验证图数据"
      width="150">
      </el-table-column>
      <el-table-column
      prop="valid_comms"
      label="验证群组数据"
      width="150">
      </el-table-column>
      <el-table-column
      prop="f1"
      label="F1"
      width="150">
      </el-table-column>
      <el-table-column
      prop="jac"
      label="JAC"
      width="150">
      </el-table-column>
      <el-table-column
      prop="create_time"
      label="时间"
      width="200">
      </el-table-column>
  </el-table>
  </div>

</template>

<script>

import {mapState, mapMutations, mapActions} from 'vuex'
export default {
  computed: {
    ...mapState('model', ['valid_results', 'titlevalid'])
  },
  
  created() {
    this.fetchValidResult()
  },
  methods: {
    ...mapActions('model', ['fetchValidResult']),
  }
}
</script>
