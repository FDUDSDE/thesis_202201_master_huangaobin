import {get} from '../../utils/axios'

export default{
    namespaced: true,
    state:{
        title: 'Graph_Data',
        list: null,
        title_c: 'Comms_Data',
        list_c: null,
    },
    // mutations:{
    //     getname(state){
    //         console.log(state.name)
    //     }
    // },
    actions:{
        fetchData(context) {
            get("/graph_data/request").then((msg)=>{
                context.state.list = msg.data;
            })
        },
        fetchDataC(context) {
            get("/comms_data/request").then((msg)=>{
                context.state.list_c = msg.data;
            })
        },
    }
}