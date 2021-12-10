<template>

  <div class="app-container">
      <h3>{{titlevalid}}</h3>
      <el-form :model="form" ref="form" label-width="100px">

        <el-form-item label="种子选择器" prop="selector">
            <el-select v-model="form.selector" placeholder="请选择种子选择器">
            <el-option v-for="item of list1" :key="item.name" :label="item.name" :value="item.name"></el-option>
            <el-option label="random" value="random"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="局部扩张器" prop='expender'>
            <el-select v-model="form.expender" placeholder="请选择训练群组">
            <el-option v-for="item of list2" :key="item.name" :label="item.name" :value="item.name"></el-option>
            <el-option label="random" value="random"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="数据集" prop="graph">
            <el-select v-model="form.graph" placeholder="请选择图数据">
            <el-option v-for="item of graph_list" :key="item.id" :label="item.graph" :value="item.graph"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="验证群组" prop='comms'>
            <el-select v-model="form.comms" placeholder="请选择验证群组">
            <el-option v-for="item of comms_list" :key="item.id" :label="item.comms" :value="item.comms"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="输出个数" prop="outputnums" style="width: 240px;">
            <el-input v-model="form.outputnums"></el-input>
        </el-form-item>

        <el-form-item>
            <el-button type="primary" @click="onSubmit">开始验证</el-button>
            <el-button @click="resetForm('form')">重置</el-button>
        </el-form-item>

    </el-form>

    <el-dialog v-el-drag-dialog :visible.sync="dialogTableVisible" title="Message" width="30%">
      <span>验证请求已发送</span>
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
        ...mapState('model', ['list1', 'list2', 'graph_list', 'comms_list', 'titlevalid'])
    },
    created () {
        this.fetchModel();
        this.fetchGraph();
        this.fetchComms();
    },
    methods: {
        ...mapActions('model', ['fetchModel', 'fetchGraph', 'fetchComms', 'modelValid']),
      onSubmit() {
        this.dialogTableVisible = true;
        this.modelValid(this.form);
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