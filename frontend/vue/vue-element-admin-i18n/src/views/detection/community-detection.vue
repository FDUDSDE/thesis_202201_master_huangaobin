<template>

  <div class="app-container">
      <h3>{{titledetection}}</h3>
      <el-form :model="form" ref="form" label-width="100px">

        <el-form-item label="种子选择器" prop="selector">
            <el-select v-model="form.selector" placeholder="请选择种子选择器">
            <el-option v-for="item of list1" :key="item.name" :label="item.name" :value="item.name"></el-option>
            </el-select>
        </el-form-item>

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

        <el-form-item label="输出个数" prop="outputnums" style="width: 240px;">
            <el-input v-model="form.outputnums"></el-input>
        </el-form-item>

        <el-form-item>
            <el-button type="primary" @click="onSubmit">开始运行</el-button>
            <el-button @click="resetForm('form')">重置</el-button>
        </el-form-item>

    </el-form>

    <el-dialog v-el-drag-dialog :visible.sync="dialogTableVisible" title="Message" width="30%">
      <span>群组发现请求已发送</span>
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
        ...mapState('detection', ['list1', 'list2', 'graph_list', 'titledetection'])
    },
    created () {
        this.fetchModel();
        this.fetchGraph();
    },
    methods: {
        ...mapActions('detection', ['fetchModel', 'fetchGraph','communityDetection']),
      onSubmit() {
        this.dialogTableVisible = true;
        this.communityDetection(this.form);
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