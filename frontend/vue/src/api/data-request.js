import Axios from "axios";
import { get } from "core-js/core/dict";
import qs from 'qs'

const myAxios = Axios.create(
    {
        baseURL: 'localhost:8080/',
        timeout: 5000,
    }
)

export function getdata() {
    return myAxios(
        {
            method: 'get',
            url: 'data/request',
            timeout: 10000.
        }
    )
}

export function postdata(data) {
    return myAxios(
        {
            method: 'post',
            url: 'data/upload',
            data: qs.stringify(data),
            timeout: 10000,
        }
    )
}