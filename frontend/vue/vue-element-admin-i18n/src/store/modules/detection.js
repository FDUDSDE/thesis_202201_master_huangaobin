import {get, postdata} from '../../utils/axios'

export default{
    namespaced: true,
    state:{
        titledetection: "Community_Detection",
        titlesearch: "Community_Search",
        list1: null,
        list2: null,
        graph_list: null,
        detection_results: null,
        search_output: null,
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
        communityDetection(context, form) {
            // console.log(form);
            postdata("/community/detection", form).then((msg)=> {
                console.log(msg.data);
            })
        },
        detectionResults(context) {
            get("/detection/request").then((msg)=>{
                context.state.detection_results = msg.data;
                // console.log(msg.data);
            })
        },
        communitySearch(context, form) {
            postdata("/community/search", form).then((msg)=> {
                context.state.search_output = msg.data.output
                console.log(msg.data);
            })
        }
    }
}