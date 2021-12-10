<template>

  <div class="app-container">
    
    <h3>{{titledetection}}</h3>
    <el-table
    :data="detection_results"
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
      prop="graph"
      label="图数据集"
      width="150">
      </el-table-column>
      <el-table-column
      prop="nums"
      label="输出个数"
      width="150">
      </el-table-column>
      <el-table-column
      prop="file"
      label="保存文件"
      width="150">
      </el-table-column>
  </el-table>
  </div>

</template>

<script>

import {mapState, mapMutations, mapActions} from 'vuex'
export default {
  computed: {
    ...mapState('detection', ['detection_results', 'titledetection'])
  },
  
  created() {
    this.detectionResults()
  },
  methods: {
    ...mapActions('detection', ['detectionResults']),
  }
}
</script>
