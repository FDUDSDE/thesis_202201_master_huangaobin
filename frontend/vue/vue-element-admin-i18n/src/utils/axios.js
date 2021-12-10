import Axios from "axios";
import qs from 'qs'

const myAxios = Axios.create(
    {
        baseURL: 'http://127.0.0.1:7000/',
        timeout: 5000,
    }
)

export function get(url) {
    return myAxios(
        {
            method: 'get',
            url: url,
            timeout: 10000.
        }
    )
}

export function postdata(url, data) {
    console.log(data);
    return myAxios(
        {
            method: 'post',
            url: url,
            data: qs.stringify(data),
            timeout: 10000,
        }
    )
}