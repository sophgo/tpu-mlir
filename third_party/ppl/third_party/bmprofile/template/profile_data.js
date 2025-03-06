// mock data for test
// required
let page_caption = "demo"

// time char required
let time_caption="Test Time Chart"
// categories are used as rows' names in the chart
let categories = ["cpu", "arm", "bdc", "gdma"]
// time_header is used as columns' names in the chart table
// the first 5 columns are required: {category_index, begin_time, end_time, type_id, height, ...}
// category_index determines the data bar is placed in which row
// begin_time and end_time determine the data bar is place in time
// type_id determines the data bar's color
// height determines the data bar's height, ranging from -1 to 1
// other fields will be used as extra info displayed in tips or used for filter
let time_header = ["device", "begin_ms", "end_ms", "type", "quality", "info"]
let time_data = [
    [2, 100,200, 1, 1,"good"],
    [1, 200,300, 2, 0.4, "good"],
    [2, 300,400, 1, 0.6,"bad"],
    [3, 400,500, 3,-1, "bad"],
    [0, 550,600, 3, 0,"bad"],
]
// tell which columns are used to filter displayed data
let filter_cols = [3, 4] 


let summary_caption = "Mock Summary"
let summary_header = ["id", "name", "age"]
let summary_data = [
    [0, "xiaotan", 20],
    [1, "lisi", 21],
    [2, "zhangsan", 20],
]
var gmem_partition = [
    // addr size desc
 [4379901952 , 16920576 ,"net_ctx->neuron_mem"],
 [4396822528 , 27361280 ,"coeff"],
 [4424183808 , 90624 ,"bd_cmd_mem"],
 [4424278016 , 175744 , "gdma_cmd_mem"],
 [4424454144 , 602112 , "tensor"],
 [4425056256 , 4 , "tensor"],
 [4425060352 , 84 , "output_mem"],
 [4425064448 , 16384 , "dyn_profile"],
 [4425080832 , 320000 , "bdc_perf_monitor"],
 [4425404416 , 1920000 , "gdma_perf_monitor"],
]

var gmem_op_record = [
    //begin end type(0:R, 1:W) addr size desc
    [ 100000, 200000, 0, 4379901952 , 1000, "tensor0"],
    [ 150000, 300000, 0, 4379901952+1000, 2000, "tensor1"],
    [ 130000, 260000, 1, 4379901952+3000, 1000, "tensor2"],
    [ 180000, 370000, 1, 4379901952+4000, 2000, "tensor3"],
]

var lmem_partition = [
    // addr size desc
 [4379901952 , 16920576 ,"net_ctx->neuron_mem"],
 [4396822528 , 27361280 ,"coeff"],
 [4424183808 , 90624 ,"bd_cmd_mem"],
 [4424278016 , 175744 , "gdma_cmd_mem"],
 [4424454144 , 602112 , "tensor"],
 [4425056256 , 4 , "tensor"],
 [4425060352 , 84 , "output_mem"],
 [4425064448 , 16384 , "dyn_profile"],
 [4425080832 , 320000 , "bdc_perf_monitor"],
 [4425404416 , 1920000 , "gdma_perf_monitor"],
]

var lmem_op_record = [
    //begin end type(0:R, 1:W) addr size desc
    [ 100000, 200000, 0, 4379901952 , 1000, "tensor0"],
    [ 150000, 300000, 0, 4379901952+1000, 2000, "tensor1"],
    [ 130000, 260000, 1, 4379901952+3000, 1000, "tensor2"],
    [ 180000, 370000, 1, 4379901952+4000, 2000, "tensor3"],
]
