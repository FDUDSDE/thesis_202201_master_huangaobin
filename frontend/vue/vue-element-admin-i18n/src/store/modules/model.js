import {get, postdata} from '../../utils/axios'

export default{
    namespaced: true,
    state:{
        title: 'Model',
        titlevalid: 'Model_Valid',
        titletrain: 'Model_Train',
        list1: null,
        list2: null,
        graph_list: null,
        comms_list: null,
        form_uoload: null,
        valid_results: null
    },
    // mutations:{
    //     getname(state){
    //         console.log(state.name)
    //     }
    // },
    actions:{
        fetchModel(context) {
            get("/model/request").then((msg)=>{
                context.state.list1 = msg.data.selectors;
                context.state.list2 = msg.data.expenders;
                // console.log(msg.data.expenders);
            })
        },
        fetchGraph(context) {
            get("/graph/request").then((msg)=>{
                context.state.graph_list = msg.data;
                // console.log(msg.data);
            })
        },
        fetchComms(context) {
            get("/comms/request").then((msg)=>{
                context.state.comms_list = msg.data;
                // console.log(msg.data);
            })
        },
        modelTrain(context, form) {
            postdata("/model/train", form).then((msg)=>{
                console.log(msg.data)
            })
        },
        modelValid(context, form) {
            postdata("/model/valid", form).then((msg)=>{
                console.log(msg.data)
            })
        },
        fetchValidResult(context) {
            get("/valid/request").then((msg)=>{
                context.state.valid_results = msg.data;
                // console.log(msg.data);
            })
        },
    }
}