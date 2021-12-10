<template>

  <div class="app-container">
      <h3>{{titlesearch}}</h3>
      <el-form :model="form" ref="form" label-width="100px">

        <el-form-item label="局部扩张器" prop='expender'>
            <el-select v-model="form.expender" placeholder="请选择训练群组">
            <el-option v-for="item of list2" :key="item.name" :label="item.name" :value="item.name"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="数据集" prop="graph">
            <el-select v-model="form.graph" placeholder="请选择图数据">
            <el-option v-for="item of graph_list" :key="item.id" :label="item.graph" :value="item.graph"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="种子输入" prop="seed" style="width: 240px;">
            <el-input v-model="form.seed"></el-input>
        </el-form-item>

        <el-form-item>
            <el-button type="primary" @click="onSubmit">开始运行</el-button>
            <el-button @click="resetForm('form')">重置</el-button>
        </el-form-item>

    </el-form>

    <el-dialog v-el-drag-dialog :visible.sync="dialogTableVisible" title="Message" width="50%">
      <span style="height:200px">种子搜索结果：{{search_output}}</span>
      <div style="height: 300px">
        </div>
      <span slot="footer" class="dialog-footer">
        <el-button type="primary" @click="finishSubmit">确 定</el-button>
      </span>
    </el-dialog>

</div>

</template>


<script>
import {mapState, mapMutations, mapActions} from 'vuex'
  export default {
    
    data() {
      return {
        dialogTableVisible:false,
        form: {
          selector: '',
          expender: '',
        },
        graph: [],
        comms: [],
      }
    },
    computed: {
        ...mapState('detection', ['list2', 'graph_list', 'titlesearch', 'search_output'])
    },
    created () {
        this.fetchModel();
        this.fetchGraph();
    },
    methods: {
        ...mapActions('detection', ['fetchModel', 'fetchGraph','communitySearch']),
      onSubmit() {
        this.dialogTableVisible = true;
        this.communitySearch(this.form);
      },
      finishSubmit() {
        this.dialogTableVisible = false;
        this.resetForm('form');
      },
      resetForm(formName) {
        this.$refs[formName].resetFields();
      },
    }
  }
</script>