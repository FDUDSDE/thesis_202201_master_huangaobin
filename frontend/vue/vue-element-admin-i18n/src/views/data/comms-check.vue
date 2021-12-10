<template>
  <div class="app-container">
    <h3>{{title_c}}</h3>

    <el-table
    :data="list_c"
    stripe
    style="width: 100%">
    <el-table-column
      prop="comms"
      label="群组数据"
      width="160">
    </el-table-column>
    <el-table-column
      prop="nums"
      label="群组数量"
      width="160">
    </el-table-column>
    <el-table-column
      prop="min_comm"
      label="最小群组szie"
      width="160">
    </el-table-column>
    <el-table-column
      prop="max_comm"
      label="最大群组size"
      width="160">
    </el-table-column>
    <el-table-column
      prop="ave_comm"
      label="平均群组size"
      width="160">
    </el-table-column>
    <el-table-column
      prop="accessable"
      label="accessable"
      width="160">
    </el-table-column>
  </el-table>

  </div>
</template>

<script>

import {mapState, mapMutations, mapActions} from 'vuex'
export default {
  computed: {
    ...mapState('data', ['title_c','list_c'])
  },

  created() {
    this.fetchDataC()
  },
  methods: {
    ...mapActions('data',['fetchDataC']),
    getList() {
      this.listLoading = false
    }
    }
}
</script>