<template>

  <div class="app-container">
    <h3>{{titletrain}}</h3>
      <el-form :model="form" ref="form" label-width="150px">

        <el-form-item label="图数据" prop="graph">
            <el-select v-model="form.graph" placeholder="请选择图数据集">
            <el-option v-for="item of graph_list" :key="item.id" :label="item.graph" :value="item.graph"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="训练群组" prop='comms'>
            <el-select v-model="form.comms" placeholder="请选择训练群组">
            <el-option v-for="item of comms_list" :key="item.id" :label="item.comms" :value="item.comms"></el-option>
            </el-select>
        </el-form-item>

        <el-form-item label="训练集大小" prop="transize" style="width: 286px;">
            <el-input v-model="form.transize"></el-input>
        </el-form-item>

        <el-form-item label="选择器训练轮数" prop="ssepochs" style="width: 286px;">
            <el-input v-model="form.ssepochs"></el-input>
        </el-form-item>

        <el-form-item label="选择器批数据大小" prop="ssbatchsize" style="width: 286px;">
            <el-input v-model="form.ssbatchsize"></el-input>
        </el-form-item>

        <el-form-item label="选择器学习率" prop="sslr" style="width: 286px;">
            <el-input v-model="form.sslr"></el-input>
        </el-form-item>

        <el-form-item label="点对数" prop="sspairs" style="width: 286px;">
            <el-input v-model="form.sspairs"></el-input>
        </el-form-item>

        <el-form-item label="扩张器训练轮数" prop="cgepochs" style="width: 286px;">
            <el-input v-model="form.cgepochs"></el-input>
        </el-form-item>

        <el-form-item label="扩张器批数据大小" prop="cgbatchsize" style="width: 286px;">
            <el-input v-model="form.cgbatchsize"></el-input>
        </el-form-item>

        <el-form-item label="扩张器学习率" prop="cglr" style="width: 286px;">
            <el-input v-model="form.cglr"></el-input>
        </el-form-item>

        <el-form-item label="ADA" prop="ada" style="width: 286px;">
            <el-input v-model="form.ada"></el-input>
        </el-form-item>

        <el-form-item>
            <el-button type="primary" @click="onSubmit">开始训练</el-button>
            <el-button @click="resetForm('form')">重置</el-button>
        </el-form-item>

    </el-form>

    <el-dialog v-el-drag-dialog :visible.sync="dialogTableVisible" title="Message" width="30%">
      <span>训练请求已发送</span>
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
          graph: '',
          comms: '',
        },
        graph: [],
        comms: [],
      }
    },
    computed: {
        ...mapState('model', ['graph_list', 'comms_list','form_upload', 'titletrain'])
    },
    created () {
        this.fetchGraph();
        this.fetchComms();
    },
    methods: {
        ...mapActions('model',['fetchGraph', 'fetchComms','modelTrain']),
      onSubmit() {
        this.dialogTableVisible = true;
        this.modelTrain(this.form);
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