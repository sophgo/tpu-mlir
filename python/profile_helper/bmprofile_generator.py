#!/usr/bin/python3
# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import os
from collections import namedtuple
from shutil import copyfile
import logging
from enum import Enum

from bmprofile_utils import *
from bmprofile_common import *
import ctypes

class ShowCategory(Enum):
    SUBNET = 5
    HOST_CPU = 4
    TPU_LAYER = 3
    NODE_OP = 2
    TPU_GDMA = 1
    TPU_BD = 0

def get_layer_type(layer, default_type="UNKNOWN", need_suffix=True):
    if layer.layer_type is not None:
        if isinstance(layer.layer_type, str):
            res = layer.layer_type
        else:
            res = layer.layer_type.name
    else:
        res = default_type if not layer.gdma_op else layer.gdma_op.name
    if need_suffix:
        local_layer_suffix = "(L)"
        global_layer_suffix = "(G)"
        res += local_layer_suffix if layer.is_local else global_layer_suffix
    return res


TimeData = namedtuple("TimeData",
                      "category begin_usec end_usec func_type height layer_id layer_type subnet_id subnet_type iteration info")

# begin_usec end_usec rw_type(0:R, 1:W) addr size desc
MemRecord = namedtuple("MemRecord", "begin_usec end_usec rw_type addr size desc")
class BMProfileGenerator:
    def __init__(self):
        pass

    def __generate_text(self, parsed_data, option):
        option = option_to_map(option)
        is_pretty = option.get("pretty", "True") == "True"

        show_summary = option.get("summary") == "True"
        show_detail= option.get("detail", "True") == "True"
        show_mem = option.get("mem") == True
        summary_data = self.__summary_data(parsed_data)
        if show_detail or show_mem:
            gmem_partition, lmem_partition = self.__partition_data(parsed_data[0])
            time_data, gmem_records, lmem_records, _ = self.__time_data(parsed_data)

        if show_summary:
            summary_header = ["index", "subnet_id", "subnet_type", "duration"]
            summary_data_out = []
            for s in summary_data:
                summary_data_out.append([s.iteration, s.subnet_id, s.subnet_type,usec_to_str(s.duration)])
            print_table("SUMMARY", summary_header, summary_data_out, is_pretty)

        if show_detail:
            detail_category = option.get("detail_category", ShowCategory.TPU_LAYER.name)
            detail_size = int(option.get("detail_size", 100))
            detail_order = option.get("detail_order", "desc")
            detail_order_by = option.get("detail_order_by", "duration").split(":")
            detail_order_by = [o.strip() for o in detail_order_by if o.strip() != ""]

            detail_filter= option.get("detail_filter", "True")
            detail_columns= option.get("detail_columns", "").split(":")
            detail_columns = [c.strip() for c in detail_columns if c.strip() != ""]
            detail_header = ["category", "begin_usec", "end_usec", "duration", "func_type",
                           "layer_id", "layer_type", "subnet_id", "subnet_type", "iteration", "info"]
            DetailItem = namedtuple("DetailItem", detail_header)
            detail_data = []
            category_timedata=[]
            for td in time_data:
                if ShowCategory(td[0]).name == detail_category:
                    category_timedata.append(td)
            for td in category_timedata:
                detail_data.append(DetailItem(
                    "{}-{}".format(td.category, ShowCategory(td.category).name),
                    td.begin_usec,
                    td.end_usec,
                    td.end_usec-td.begin_usec,
                    td.func_type,
                    td.layer_id,
                    td.layer_type,
                    td.subnet_id,
                    td.subnet_type,
                    td.iteration,
                    td.info.replace("\n", ",").replace("<br>",",").strip(",")
                ))
            if len(detail_columns) == 0:
                detail_columns = detail_header

            out_items = filter_items(detail_data, detail_filter, detail_order_by, detail_order != "asc", detail_size, detail_columns)
            time_columns = ["begin_usec", "end_usec", "duration"]
            time_indice = [ i for i,c in enumerate(detail_columns) if c in time_columns ]
            if len(time_indice)>0:
                for item in out_items:
                    for t in time_indice:
                        item[t] = usec_to_str(item[t])
            print_table("TIME DATA", detail_columns, out_items, is_pretty)
            print("total {} items, show top {} items for {}".format(len(category_timedata), len(out_items), detail_category))

        # if show_mem:
        #     mem_size = int(option.get("mem_size", 100))
        #     mem_order = option.get("mem_order", "asc")
        #     mem_order_by = option.get("mem_order_by", "begin_usec").split(":")
        #     mem_filter= option.get("mem_filter", "")
        #     mem_columns= option.get("mem_columns", None)

    def __generate_csv(self, parsed_data, out_dir):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        assert os.path.isdir(out_dir)

        summary_data = self.__summary_data(parsed_data)
        summary_file = os.path.join(out_dir, "summary.csv")
        item_config={"sep":",", "suffix":"\n", "prefix":""}
        with open(summary_file, "w") as f:
            f.write(self.__item_str(["index", "subnet_id", "subnet_type", "duration(us)"], **item_config))
            for s in summary_data:
                f.write(self.__item_str((s.iteration, s.subnet_id, s.subnet_type, s.duration), **item_config))

        gmem_partition, lmem_partition = self.__partition_data(parsed_data[0])
        time_data, gmem_records, lmem_records, _ = self.__time_data(parsed_data)

        detail_file = os.path.join(out_dir, "detail.csv")
        mem_file = os.path.join(out_dir, "mem.csv")
        with open(detail_file, "w") as f:
            category_num = len(ShowCategory.__members__.keys())
            categories = [ShowCategory(c).name for c in range(category_num)]
            # f.write('let categories = {}\n'.format(self.__item_str(categories)))
            time_header = ["category", "begin_usec", "end_usec", "duration(us)", "func_type",
                           "layer_id", "layer_type", "subnet_id", "subnet_type", "iteration", "info"]
            f.write(self.__item_str(time_header, **item_config))
            for td in time_data:
                f.write(self.__item_str((
                    "{}-{}".format(td.category, ShowCategory(td.category).name),
                    td.begin_usec,
                    td.end_usec,
                    td.end_usec-td.begin_usec,
                    td.func_type,
                    td.layer_id,
                    td.layer_type,
                    td.subnet_id,
                    td.subnet_type,
                    td.iteration,
                    td.info.replace("\n", ",").replace("<br>",",").strip(",")
                ), **item_config))

        with open(mem_file, "w") as f:
            mem_header = [ "mem_type", "start_addr", "end_addr", "byte_size", "rw_type", "start_usec", "end_usec", "duration", "partition", "info"]
            f.write(self.__item_str(mem_header, **item_config))
            def write_records(type_name, records, partitions):
                for m in records:
                    note = ""
                    for p in partitions:
                        if m.addr >= p[0] and m.addr < p[0]+p[1]:
                            note = p[2]
                            break
                    f.write(self.__item_str((
                        type_name,
                        m.addr,
                        m.addr+m.size,
                        m.size,
                        "R" if m.rw_type==0 else "W",
                        m.begin_usec,
                        m.end_usec,
                        m.end_usec-m.begin_usec,
                        note,
                        m.desc
                    ), **item_config))
            write_records("global", gmem_records, gmem_partition)
            write_records("local", lmem_records, lmem_partition)

    def __generate_html(self, parsed_data, out_dir):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        assert os.path.isdir(out_dir)
        source_dir = os.path.join(os.path.dirname(__file__), "template")
        file_to_copy = ["result.html", "echarts.min.js"]
        for f in file_to_copy:
            source = os.path.join(source_dir, f)
            target = os.path.join(out_dir, f)
            try:
                copyfile(source, target)
            except IOError as e:
                logging.fatal("Unable to copy file. %s" % e)
                exit(1)
            except:
                logging.fatal("Unexpected error:", sys.exc_info())
                exit(1)
        summary_data = self.__summary_data(parsed_data)
        data_filepath = os.path.join(out_dir, "profile_data.js")
        global_data = parsed_data[0]

        page_caption = "Profile: "
        if global_data.net_name:
            page_caption += global_data.net_name + " on "
        page_caption += global_data.archlib.arch_name
        if global_data.no_perf_data:
            page_caption += " (simulated)"
        with open(data_filepath, "w") as f:
            f.write('let page_caption="{}"\n'.format(page_caption))
            f.write('let summary_caption= "Subnet Summary"\n')
            f.write('let summary_header = ["index", "subnet_id", "subnet_type", "duration"]\n')
            f.write("let summary_data = [\n")
            for s in summary_data:
                f.write("  {},\n".format(self.__item_str((s.iteration, s.subnet_id, s.subnet_type, usec_to_str(s.duration)))))
            f.write("]\n")
            category_num = len(ShowCategory.__members__.keys())
            categories = [ShowCategory(c).name for c in range(category_num)]
            f.write('let categories = {}\n'.format(self.__item_str(categories)))
            time_header = TimeData._fields
            filter_cols = [time_header.index(c) for c in ["layer_id", "layer_type", "subnet_id", "subnet_type", "iteration"]]
            f.write('let time_header = {}\n'.format(self.__item_str(time_header)))
            f.write('let filter_cols={}\n'.format(self.__item_str(filter_cols)))
            gmem_partition, lmem_partition = self.__partition_data(parsed_data[0])
            self.__write_data(f, "lmem_partition", lmem_partition)
            self.__write_data(f, "gmem_partition", gmem_partition)
            time_data, gmem_records, lmem_records, _ = self.__time_data(parsed_data)
            self.__write_data(f, "time_data", time_data)
            self.__write_data(f, "gmem_op_record", gmem_records)
            self.__write_data(f, "lmem_op_record", lmem_records)
        return os.path.join(out_dir, file_to_copy[0])

    def __write_data(self, f, var_name, data):
            f.write("let {}= [\n".format(var_name))
            for d in data:
                f.write("  {},\n".format(self.__item_str(d)))
            f.write("]\n")

    def __partition_data(self, global_data):
        archlib = global_data.archlib
        lmem_partition = archlib.local_mem_partition()
        gmem_partition = []
        for m in global_data.mem_info:
            gmem_partition.append((m.addr, m.size, m.desc))
        return gmem_partition, lmem_partition

    def __append_extra(self, info, extra_data, profile_id):
        extra_items = extra_data.get(profile_id, [])
        extra = []
        for e in extra_items:
            if e.profile_id == profile_id and e.type == DynExtraType.STRING:
                extra.append(str(e.content))
        if len(extra) >0:
            info.append("extra=" + (",".join(extra)))

    def __device_time_data(self, idata, global_data):

        archlib = global_data.archlib

        time_data = []
        gmem_records = []
        lmem_records = []
        # used to fix perf_monitor start time
        summary = idata.parsed_summary

        gdma_monitor_data=[]
        gdma_set_data = []
        bd_set_data = []
        subnet_info = idata.subnet_info
        gdma_nodes = []
        bd_nodes = []
        layer_info = []
        dyn_begin_usec = 0
        if subnet_info is not None:
            layer_info = subnet_info.layer_list
            gdma_nodes, bd_nodes = subnet_info.gdma_nodes, subnet_info.bd_nodes
        if idata.dyn_data is not None and len(idata.dyn_data)>=2:
            #the first record tells how many nanoseconds in a cycle
            current_func = None
            dyn_begin_usec = summary.begin_usec

            # hack here: the following code should be REMOVED
            # but currently there is an unknown time shift for bmlib
            if len(summary.send_info)>0:
                dyn_begin_usec = summary.send_info[0].end_usec + 1
                for s in summary.send_info:
                    if BMLibApi.will_run_kernel(s.api):
                        dyn_begin_usec = s.end_usec + 1
                        break

            layer_id = -1
            layer_type = "-"

            for i in range(1, len(idata.dyn_data)):
                raw_dyn = idata.dyn_data[i]
                dyn_type = archlib.DynRecordType(raw_dyn.type)
                profile_id = raw_dyn.profile_id
                raw_id = raw_dyn.id

                category = ShowCategory.NODE_OP.value
                if dyn_type == archlib.DynRecordType.FUNC:
                    category = ShowCategory.TPU_LAYER.value
                begin_usec = dyn_begin_usec + raw_dyn.begin_usec
                end_usec = dyn_begin_usec + raw_dyn.end_usec
                subnet_id = summary.subnet_id
                subnet_type = summary.subnet_type
                iteration = summary.iteration

                func_type, height, layer_id, layer_type, info, bd_set_data, gdma_set_data = archlib.parse_raw_id(dyn_type, raw_id, begin_usec, end_usec, current_func, bd_set_data, gdma_set_data, layer_id, layer_type)

                if raw_dyn.info:
                    info += raw_dyn.info
                if dyn_type != archlib.DynRecordType.FUNC and current_func is not None and begin_usec>=current_func.begin_usec and end_usec <= current_func.end_usec:
                    layer_id = current_func.layer_id
                    layer_type = current_func.layer_type
                time_data.append(TimeData(
                    category = category,
                    begin_usec = begin_usec,
                    end_usec = end_usec,
                    func_type = func_type,
                    height = height,
                    layer_id = layer_id,
                    layer_type = layer_type,
                    subnet_id = subnet_id,
                    subnet_type = subnet_type,
                    iteration = iteration,
                    info = ", ".join(info),
                    ))
                if dyn_type == archlib.DynRecordType.FUNC:
                    current_func = time_data[-1]

        def collect_mem_records(m, gmem_records, lmem_records):
            if not m:
                return
            if not m.dynamic and not m.static:
                return
            n = m.dynamic if m.dynamic else m.static
            if n.command and n.command.mem_records:
                for r in n.command.mem_records:
                    # begin end type(0:R, 1:W) addr size desc
                    record = MemRecord(
                        begin_usec = begin_usec,
                        end_usec = end_usec,
                        rw_type = r[4], addr = r[0],
                        size = r[2],
                        desc = r[5])
                    if r[3]:
                        gmem_records.append(record)
                    else:
                        lmem_records.append(record)
        if idata.monitor_gdma is not None and len(idata.monitor_gdma)>0:
            gdma_trans_data = []
            max_trans_speed = 0
            for m in idata.monitor_gdma:
                if m.d0_wr_bytes == 0xFFFFFFFF or m.d1_wr_bytes==0xFFFFFFFF or m.gif_wr_bytes == 0xFFFFFFFF:
                    continue
                begin_usec = m.inst_start_time* global_data.gdma_period + dyn_begin_usec
                end_usec = m.inst_end_time * global_data.gdma_period + dyn_begin_usec
                gdma_monitor_data.append([m.inst_id, begin_usec, end_usec])
                trans_bytes = m.d0_wr_bytes + m.d1_wr_bytes + m.gif_wr_bytes
                trans_speed = trans_bytes/(end_usec-begin_usec) if end_usec> begin_usec else 0
                if max_trans_speed<trans_speed:
                    max_trans_speed = trans_speed
                info = "speed=%.3fGB/s, bytes=%d"%(trans_speed/1000, trans_bytes)
                info += "<br>d0_wr={}, d0_ar={}, d0_aw={}".format(m.d0_wr_bytes, m.d0_ar_bytes, m.d0_aw_bytes)
                info += "<br>d1_wr={}, d1_ar={}, d1_aw={}".format(m.d1_wr_bytes, m.d1_ar_bytes, m.d1_aw_bytes)
                info += "<br>gif_wr={}, gif_ar={}, gif_aw={}".format(m.gif_wr_bytes, m.gif_ar_bytes, m.gif_aw_bytes)
                # info += "<br>"+str(m).replace(" ", "<br>")

                layer_type = "-"
                layer_id = -1
                collect_mem_records(m, gmem_records, lmem_records)
                if m.static:
                    n = m.static
                    if n.layer is not None:
                        n.layer.update_time(begin_usec, end_usec)
                        layer_id = n.layer.layer_id
                        layer_type = get_layer_type(n.layer)
                    info = "{}<br>".format(n.gdma_func.name) + info
                time_data.append(TimeData(
                    category = ShowCategory.TPU_GDMA.value,
                    begin_usec = begin_usec,
                    end_usec = end_usec,
                    func_type = "gdma_id={}".format(m.inst_id),
                    height = trans_speed/max_trans_speed,
                    layer_id =  layer_id,
                    layer_type = layer_type,
                    subnet_id = summary.subnet_id,
                    subnet_type = summary.subnet_type,
                    iteration = summary.iteration,
                    info = info,
                    ))

        if idata.monitor_bd is not None and len(idata.monitor_bd)>0:
            for idx, m in enumerate(idata.monitor_bd):
                begin_usec = m.inst_start_time*global_data.tiu_period + dyn_begin_usec
                end_usec = m.inst_end_time*global_data.tiu_period + dyn_begin_usec
                info = "cycle=%d"%(m.computation_load)
                if len(bd_set_data) > 0 and idx < len(bd_set_data):
                    info += "<br>%s"%(bd_set_data[idx][-1])

                layer_type = "-"
                layer_id = -1
                collect_mem_records(m, gmem_records, lmem_records)
                height = 1
                if m.command and hasattr(m.command, "alg_ops"):
                    alg_ops = m.command.alg_ops
                    arch_ops = m.command.arch_ops
                    height = alg_ops/arch_ops if arch_ops and arch_ops>0 else 1
                    info += "<br>ops_ratio={}".format(height)

                if m.static:
                    n = m.static
                    if n.layer is not None:
                        n.layer.update_time(begin_usec, end_usec)
                        layer_id = n.layer.layer_id
                        layer_type = get_layer_type(n.layer, "-")
                        info = n.bd_func.name+"<br>"+ info
                time_data.append(TimeData(
                    category = ShowCategory.TPU_BD.value,
                    begin_usec = begin_usec,
                    end_usec = end_usec,
                    func_type = "bd_id={}".format(m.inst_id),
                    height = height,
                    layer_id =  layer_id,
                    layer_type = layer_type,
                    subnet_id = summary.subnet_id,
                    subnet_type = summary.subnet_type,
                    iteration = summary.iteration,
                    info = info
                    ))

        subnet_timeoffset = 0
        first_begin_usec = min([summary.begin_usec]+[layer.begin_usec for layer in layer_info if layer.begin_usec is not None])
        summary_begin_usec = summary.begin_usec
        if len(summary.send_info)>0:
            summary_begin_usec = summary.send_info[0].end_usec + 1
        if first_begin_usec is not None and first_begin_usec < summary_begin_usec:
            subnet_timeoffset = summary_begin_usec-first_begin_usec

        for layer in layer_info:
            if layer.begin_usec is None:
                continue
            layer_type = get_layer_type(layer)
            height = 1
            if layer.layer_id == -1 or (layer_type in ['Load', 'Store', 'Load(L)', 'Store(L)']):
                height = 0.5
            timeoffset = 0
            time_data.append(TimeData(
                category = ShowCategory.TPU_LAYER.value,
                begin_usec = layer.begin_usec + subnet_timeoffset,
                end_usec = layer.end_usec + subnet_timeoffset,
                func_type = layer_type,
                height = height,
                layer_id =  layer.layer_id,
                layer_type = layer_type,
                subnet_id = summary.subnet_id,
                subnet_type = summary.subnet_type,
                iteration = summary.iteration,
                info = layer.info(),
                ))
        if global_data.no_perf_data:
            for gdma_node  in gdma_nodes:
                sim_info = gdma_node.sim_info
                if sim_info is None:
                    continue
                info = sim_info.op_type+"<br>direction=" + str(sim_info.direction) +"<br>bytes=" + str(sim_info.byte_size) +"<br>speed=" + str(sim_info.bandwidth) + "GB/s"
                layer = gdma_node.layer
                layer_type = "Unknown"
                layer_id = -1
                if layer is not None:
                    layer_type = get_layer_type(layer)
                    layer_id = layer.layer_id
                time_data.append(TimeData(
                    category = ShowCategory.TPU_GDMA.value,
                    begin_usec = sim_info.start_time + subnet_timeoffset,
                    end_usec = sim_info.end_time + subnet_timeoffset,
                    func_type = "gdma_id={}".format(sim_info.gdma_id),
                    height = sim_info.bandwidth/64.0,
                    layer_id =  layer_id,
                    layer_type = layer_type,
                    subnet_id = summary.subnet_id,
                    subnet_type = summary.subnet_type,
                    iteration = summary.iteration,
                    info = info,
                    ))
            for bd_node in bd_nodes:
                sim_info = bd_node.sim_info
                if sim_info is None:
                    continue
                info = sim_info.op_type
                layer = bd_node.layer
                layer_type = "Unknown"
                layer_id = -1
                if layer is not None:
                    layer_type = get_layer_type(layer, '-')
                    layer_id = layer.layer_id
                time_data.append(TimeData(
                    category = ShowCategory.TPU_BD.value,
                    begin_usec=sim_info.start_time + subnet_timeoffset,
                    end_usec=sim_info.end_time + subnet_timeoffset,
                    func_type = "bd_id={}".format(sim_info.bd_id),
                    height = -1,
                    layer_id =  layer_id,
                    layer_type = layer_type,
                    subnet_id = summary.subnet_id,
                    subnet_type = summary.subnet_type,
                    iteration = summary.iteration,
                    info = info
                    ))
        return time_data, gmem_records, lmem_records, layer_info

    def __time_data(self, parsed_data):
        global_data, iter_data, bmlib_data = parsed_data
        all_data = iter_data + bmlib_data
        time_data = []
        # add iteration info
        for data in all_data:
            s = data.parsed_summary
            time_data.append(
                TimeData(
                    category = ShowCategory.SUBNET.value,
                    begin_usec = s.begin_usec,
                    end_usec = s.end_usec,
                    func_type = s.iteration,
                    height = -1,
                    layer_id = -1,
                    layer_type = "-",
                    subnet_id = s.subnet_id,
                    subnet_type = s.subnet_type,
                    iteration = s.iteration,
                    info = s.info))
            all_info = s.send_info + s.sync_info + s.mark_info + s.copy_info + s.mem_info
            all_info = sorted(all_info)
            for i, sinfo in enumerate(all_info):
                func_type = sinfo.func_type
                if i>0 and sinfo.func_type == "thread_sync" and all_info[i-1].api is not None:
                    func_type = "sync:"+all_info[i-1].func_type
                time_data.append(TimeData(
                    category = ShowCategory.HOST_CPU.value,
                    begin_usec = sinfo.begin_usec,
                    end_usec = sinfo.end_usec,
                    func_type = func_type,
                    height = -1,
                    layer_id = -1,
                    layer_type = "-",
                    subnet_id = -1,
                    subnet_type = s.subnet_type,
                    iteration = s.iteration,
                    info = sinfo.info))

        # add dynamic/static TPU subnet info
        gmem_records = []
        lmem_records = []
        layer_info = []
        for idata in all_data:
            device_time_data, device_gmem_records, device_lmem_records, device_layer_info =  self.__device_time_data(idata, global_data)
            time_data += device_time_data
            gmem_records += device_gmem_records
            lmem_records += device_lmem_records
            layer_info += device_layer_info
        return time_data, gmem_records, lmem_records, layer_info

    def __item_str(self, item, sep=",", prefix="[", suffix="]"):
        line = prefix
        for v in item:
            if type(v) == str:
                v='"'+v+'"'
            elif isinstance(v, int) or isinstance(v, float):
                pass
            else:
                v='"'+str(v)+'"'

            line += '{}'.format(v)
            line+= sep
        line+= suffix
        return line

    def __summary_data(self, parsed_data):

        SummaryData = namedtuple("SummaryData",
            "iteration subnet_id subnet_type begin_usec end_usec duration sync_info send_info mark_info copy_info mem_info info")
        ExtraInfo = namedtuple("ExtraInfo", "begin_usec end_usec func_type info api")

        global_data, iter_data, bmlib_data = parsed_data
        subnet_list = global_data.subnet_list
        subnet_map = { s.subnet_id: s for s in subnet_list}

        summary_data  = []

        if global_data.no_perf_data:
            last_end = 0
            for idx, idata in enumerate(iter_data):
                gsubnet = global_data.subnet_list[idx]
                summary = idata.summary
                if gsubnet.sim_info is not None:
                    begin_usec = gsubnet.sim_info[0] + last_end
                    end_usec = gsubnet.sim_info[1] + last_end
                    if isinstance(begin_usec,float):
                        begin_usec = int(begin_usec)
                    if isinstance(end_usec,float):
                        end_usec = int(end_usec)
                    summary.begin_usec = begin_usec
                    summary.end_usec = end_usec
                elif idx>0:
                    cost_time = summary.end_usec - summary.begin_usec
                    summary.begin_usec = iter_data[idx-1].summary.end_usec
                    summary.end_usec = iter_data[idx-1].summary.end_usec + cost_time
                last_end = summary.end_usec

        # calculate the minimium begin time for the whole profile
        iter_begin_time = None if len(iter_data) == 0 else iter_data[0].summary.begin_usec
        bmlib_begin_time = None if len(bmlib_data) == 0 else bmlib_data[0].summary.begin_usec
        begin_time = None
        if iter_begin_time is not None and bmlib_begin_time is not None:
            begin_time = min(iter_begin_time, bmlib_begin_time)
        elif iter_begin_time is not None:
            begin_time = iter_begin_time
        elif bmlib_begin_time is not None:
            begin_time = bmlib_begin_time
        else:
            begin_time = 0

        time_shift = 0 # to remove profile processing time

        for idx, idata in enumerate(iter_data):
            summary = idata.summary
            if idx==0:
                time_shift = 0
            else:
                time_shift = time_shift + summary.begin_usec - iter_data[idx-1].summary.end_usec
            if summary is None:
                continue
            subnet_type = enum_cast(summary.subnet_type, SubnetType)
            type_name = subnet_type.name
            if (subnet_type == SubnetType.CPU):
                cpu_type_name = enum_cast(summary.extra_data, CPULayerType).name
                type_name += "({})".format(cpu_type_name)
            elif(subnet_type == SubnetType.TPU):
                if bool(summary.extra_data):
                    type_name += "(dynamic)"
                else:
                    type_name += "(static)"
            elif(subnet_type == SubnetType.SWITCH):
                type_name += "({})".format(str(bool(summary.extra_data)))

            info = ""
            subnet_info = subnet_map.get(summary.subnet_id, None)
            if subnet_info is not None:
                info = "num_layers=%d"%(len(subnet_info.layer_list))

            begin_usec = summary.begin_usec - begin_time - time_shift
            end_usec = summary.end_usec - begin_time - time_shift
            mark_info = []
            if subnet_type != SubnetType.TPU:
                mark_info.append(ExtraInfo(
                    begin_usec = begin_usec,
                    end_usec = end_usec,
                    func_type = subnet_type.name,
                    info = "",
                    api = None,
                ))
            idata.parsed_summary = SummaryData(
                iteration="Iter[{}]".format(summary.iteration),
                subnet_id=summary.subnet_id,
                subnet_type=type_name,
                duration= end_usec - begin_usec,
                begin_usec = begin_usec,
                end_usec = end_usec,
                sync_info=[],
                send_info=[],
                mark_info= mark_info,
                copy_info=[],
                mem_info=[],
                info = info,
                )
            summary_data.append(idata.parsed_summary)

        for idx, bdata in enumerate(bmlib_data):
            summary = bdata.summary
            if idx==0:
                time_shift = 0
            elif summary.begin_usec>bmlib_data[idx-1].summary.end_usec:
                time_shift = time_shift + summary.begin_usec - bmlib_data[idx-1].summary.end_usec
            sync_info = []
            send_info = []
            mark_info = []
            copy_info = []
            mem_info = []
            for i in summary.sync_info:
                sync_info.append(ExtraInfo(
                    begin_usec = i.begin_usec - begin_time - time_shift,
                    end_usec = i.end_usec - begin_time - time_shift,
                    func_type = "thread_sync",
                    info = i.info,
                    api = None,
                ))
            for i in summary.send_info:
                send_info.append(ExtraInfo(
                    begin_usec = i.begin_usec - begin_time - 1 - time_shift,
                    end_usec = i.begin_usec - begin_time - time_shift,
                    func_type = i.info if i.info else i.api.name,
                    info = i.info,
                    api = i.api,
                ))
            for i in summary.mark_info:
                begin_usec = i.begin_usec - begin_time
                end_usec = i.end_usec - begin_time
                info = "interval"
                if i.end_usec == 0:
                    begin_usec = i.begin_usec - begin_time - 1
                    end_usec = i.begin_usec - begin_time
                    info = "point"
                mark_info.append(ExtraInfo(
                    begin_usec = begin_usec - time_shift,
                    end_usec = end_usec - time_shift,
                    func_type = "mark:id={}".format(i.mark_id),
                    info = info + ": " + i.info,
                    api = None,
                ))
            for i in summary.copy_info:
                copy_info.append(ExtraInfo(
                    begin_usec = i.begin_usec - begin_time - time_shift,
                    end_usec = i.end_usec - begin_time - time_shift,
                    func_type = "MEMCPY_"+enum_cast(i.dir, BMLibMemDir).name,
                    info = "src=0x%0lx, dst=0x%0lx, size=%d, bandwidth=%s"%(i.src_addr, i.dst_addr, i.size, calc_bandwidth(i.size, i.end_usec-i.begin_usec)),
                    api = None,
                ))
            for i in summary.mem_info:
                sync_info.append(ExtraInfo(
                    begin_usec = i.begin_usec - begin_time - time_shift,
                    end_usec = i.end_usec - begin_time - time_shift,
                    func_type = "MEM_"+enum_cast(i.type, BMLibMemOpType).name,
                    info = "addr=0x%0lx, size=%d "%(i.device_addr, i.size) + i.info,
                    api = None,
                ))
            bdata.parsed_summary = SummaryData(
                iteration=summary.iteration,
                subnet_id=-1,
                subnet_type="bmlib",
                duration=summary.end_usec - summary.begin_usec,
                begin_usec=summary.begin_usec - begin_time - time_shift,
                end_usec=summary.end_usec - begin_time - time_shift,
                sync_info=sync_info,
                send_info=send_info,
                mark_info=mark_info,
                copy_info=copy_info,
                mem_info=mem_info,
                info="",
                )
            summary_data.append(bdata.parsed_summary)
        return summary_data

    def generate(self, parsed_data, out_dir, out_format, option):
        if out_format.lower() == "html":
            result_path = self.__generate_html(parsed_data, out_dir)
            try:
                import webbrowser
                webbrowser.open(result_path)
            except:
                logging.info("Please open '{}' with your web browser.")
        elif out_format.lower() == "console":
            result_path = self.__generate_text(parsed_data, option)
        elif out_format.lower() == "csv":
            result_path = self.__generate_csv(parsed_data, out_dir)
        elif out_format.lower() == "layer":
            result_path = self.__generate_layer(parsed_data, out_dir)
        else:
            logging.fatal("unsupported format: {}".format(out_format))

    def __write_csv(self, file_name, header, data):
        item_config={"sep":",", "suffix":"\n", "prefix":""}
        with open(file_name, "w") as f:
            f.write(self.__item_str(header, **item_config))
            for d in data:
                f.write(self.__item_str(d, **item_config))
        print("file generated: "+ file_name)

    def __bmlib_layers(self, bmlib_data):

        def fake_layer(raw_layer):
            layer.layer_type = layer_type
            layer.layer_id = layer_id
            layer.is_local = False
            layer.in_tensors = []
            layer.out_tensors = []
            layer.io_info = lambda: ""
            return raw_layer

        bmlib_layers = []
        for item in bmlib_data:
            all_nodes = item.monitor_bd + item.monitor_gdma
            if all_nodes:
                all_nodes = sorted(all_nodes, key = lambda n: n.inst_start_time)
                for n in all_nodes:
                    layer_type = ""
                    layer_id = item.summary.iteration
                    if n.dynamic:
                        d = n.dynamic
                        if hasattr(d, "layer"):
                            layer = d.layer
                            layer_type = layer.info
                            layer_id = layer_id + "-" + str(d.layer.layer_id)
                            if layer not in bmlib_layers:
                                bmlib_layers.append(fake_layer(layer))
                        elif hasattr(d, "wait"):
                            layer = d.wait
                            layer_type = "unknown"
                            layer_id ="wait" + str(d.wait.end_usec)
                            if layer not in bmlib_layers:
                                bmlib_layers.append(fake_layer(layer))
        return bmlib_layers

    def __generate_layer(self, parsed_data, out_dir):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        assert os.path.isdir(out_dir)
        global_data, _, bmlib_data = parsed_data
        archlib = global_data.archlib
        BDPeriod = archlib.BDCyclePeriod if global_data.freq is None or archlib.PeriodFixed else 1.0/global_data.freq
        GDMAPeriod = archlib.GDMACyclePeriod if global_data.freq is None or archlib.PeriodFixed else 1.0/global_data.freq

        _ = self.__summary_data(parsed_data)
        _, _, _, layer_info = self.__time_data(parsed_data)

        if bmlib_data:
            layer_info += self.__bmlib_layers(bmlib_data)

        def get_inst_param(cmd):
            fields = cmd.attr_names()
            info = []
            for k in fields:
                v = cmd.__dict__[k]
                info.append(f'{k}={v}')
            return ",".join(info)

        def get_tiu_param(cmd, keys):
            values = [""]*len(keys)
            for i, k in enumerate(keys):
                k = k[4:] # remove 'des_'
                v = cmd.__dict__.get(k, "-")
                values[i] = v
            return values

        def get_dma_param(cmd, keys):
            _def = {
                "des_res0_addr": cmd.dst_start_addr_h8 * (2**32)
                + cmd.dst_start_addr_l32,
                "des_res0_n": ("dst_nsize",),
                "des_res0_c": ("dst_csize",),
                "des_res0_h": ("dst_hsize",),
                "des_res0_w": ("dst_wsize",),
                "des_res0_n_str": ("dst_nstride",),
                "des_res0_c_str": ("dst_cstride",),
                "des_res0_h_str": ("dst_hstride",),
                "des_res0_w_str": ("dst_wstride",),
                "des_opd0_addr": cmd.src_start_addr_h8 * (2**32)
                + cmd.src_start_addr_l32,
                "des_opd0_n": ("src_nsize",),
                "des_opd0_c": ("src_csize",),
                "des_opd0_h": ("src_hsize",),
                "des_opd0_w": ("src_wsize",),
                "des_opd0_n_str": ("src_nstride",),
                "des_opd0_c_str": ("src_cstride",),
                "des_opd0_h_str": ("src_hstride",),
                "des_opd0_w_str": ("src_wstride",),
                "des_opd0_prec": ("src_data_format",),
            }

            values = [""] * len(keys)
            for i, k in enumerate(keys):
                v = "-"
                if k in _def:
                    if isinstance(_def[k], int):
                        values[i] = _def[k]
                        continue
                    for _k in _def[k]:
                        if _k in cmd.__dict__:
                            v = cmd.__dict__.get(_k)
                            break
                values[i] = v
            return values

        def get_dma_direction(cmd):
            if cmd.mlir_cmd:
                mem_type = {
                    "R": "lmem",
                    "G": "ddr",
                    "S": "static",
                    "L": "l2sram",
                    "C": "const",
                }
                try:
                    dst, src = cmd.mlir_cmd.results[0], cmd.mlir_cmd.operands[0]
                    src_str = src.mtype.name if hasattr(src, "mtype") else "C"
                    dst_str = dst.mtype.name if hasattr(dst, "mtype") else "C"
                    return f"{mem_type[src_str]}->{mem_type[dst_str]}"
                except:
                    print("-----> failed to get direction!")
                    print(cmd.mlir_cmd)
                    print(cmd.mlir_cmd.reg)
            return "UNKNOWN"

        def get_inst_io_dtype(cmd):
            if cmd.mlir_cmd:
                dst, src = cmd.mlir_cmd.results[0], cmd.mlir_cmd.operands[0]
                return (src.dtype.name, dst.dtype.name)
            return ("", "")

        def get_layer_dtype(layer):
            if layer.in_tensors:
                return {layer.in_tensors[0].dtype.name}
            if layer.out_tensors:
                return {layer.out_tensors[0].dtype.name}
            dtypes = set()
            all_nodes = layer.bd_nodes + layer.gdma_nodes
            for n in all_nodes:
                if not n.pmu_info or not n.pmu_info.command:
                    continue
                if not hasattr(n.pmu_info.command, "mlir_cmd"):
                    continue
                cmd = n.pmu_info.command.mlir_cmd
                if not cmd:
                    continue
                for o in cmd.operands:
                    dtypes.add(o.dtype.name)
                for o in cmd.results:
                    dtypes.add(o.dtype.name)
            return dtypes

        def get_layer_nchw(layer, is_in):
            shape = [1,1,1,1]
            tensors = layer.in_tensors if is_in else layer.out_tensors
            if tensors:
                for i, s in enumerate(tensors[0].shape):
                    if i<4:
                        shape[i] = s
                    else:
                        shape[3] * s
            return shape

        def get_layer_bytes(layer, is_in):
            total_bytes = 0
            tensors = layer.in_tensors if is_in else layer.out_tensors
            for tensor in tensors:
                tensor_bytes = get_dtype_size(tensor.dtype)
                for s in tensor.shape:
                    tensor_bytes *= s
                total_bytes += tensor_bytes
            return total_bytes
        def get_layer_weight_size(layer):
            total_bytes = 0
            tensors = layer.in_tensors
            for tensor in tensors:
                if not tensor.is_const:
                    continue
                tensor_bytes = get_dtype_size(tensor.dtype)
                for s in tensor.shape:
                    tensor_bytes *= s
                total_bytes += tensor_bytes
            return total_bytes
        def get_ratio_str(x, y):
            return '%.3f%%'%(x/y*100) if y != 0 else "--"

        inst_header = ["EngineId", "Inst Function", "DMA data size(B)", "Direction", "Opd0 Dtype", "Res Dtype", "Layer ID", "Alg cycle", "1684x Cycle", "StartCycle", "EndCycle", "Inst ID", "Dep ID", "pmu_info", "Alg Ops", "uArch Ops", "uArch Rate",
                       "des_res0_n", "des_res0_c", "des_res0_h", "des_res0_w", "des_opd0_n", "des_opd0_c", "des_opd0_h", "des_opd0_w", "des_opd1_n", "des_opd1_c", "des_opd1_h", "des_opd1_w", "des_res0_n_str", "des_res0_c_str", "des_opd0_n_str",
                       "des_opd0_c_str", "des_opd1_n_str", "des_opd1_c_str", "des_opd2_n_str", "des_opd2_c_str", "des_res0_addr", "des_opd0_addr", "des_opd1_addr", "des_opd2_addr", "des_res0_h_str", "des_res0_w_str", "des_opd0_h_str", "des_opd0_w_str", "des_opd1_h_str", "des_opd1_w_str", "des_opd2_h_str", "des_opd2_w_str", "des_res1_addr", "des_opd3_addr", "des_res_op_x_str", "des_res_op_y_str", "des_opd0_prec", "des_res0_prec"]
        reg_begin_index = inst_header.index("des_res0_n")
        layer_header = ["Layer ID", "Type", "TPU/CPU", "DataType", "Function",
                        "Load Bandwidth(GBps)", "Store Bandwidth(GBps)",
                        "kh", "kw", "KStrideH", "KStrideW", "Padding", "in", "ic", "ih", "iw", "on", "oc", "oh", "ow",
                        "ifm(B)", "ofm(B)", "weight(B)",
                        "AlgorithmOps", "Other info",
                        "uArch URate", "uArch Ops", "uArch CModel Cycle", "uArch Cycle Ratio",
                        "1684x Cycle", "1684x Time Ratio"]

        layer_id_map = dict()
        inst_data = []
        null_regs = [""] * (len(inst_header) - reg_begin_index)
        total_tiu_cycles = 0
        total_arch_cycles = 0
        total_arch_ops = 0
        total_alg_ops = 0
        for layer in layer_info:
            layer_type = get_layer_type(layer, "", False)
            layer_alg_ops = 0
            layer_arch_ops = 0
            conv_kh = 0
            conv_kw = 0
            conv_stride_h = 0
            conv_stride_w = 0
            conv_pad = [0, 0, 0, 0]
            for n in layer.bd_nodes:
                if not n.pmu_info:
                    continue
                command_type  = n.pmu_info.command.type if n.pmu_info.command else "UNKNOWN"
                inst_id = f'TIU-{n.pmu_info.inst_id}'
                dep_id = f'GDMA-{n.pmu_info.command.dep_id if n.pmu_info.command else "UNKNOWN"}'
                arch_cycles = int(1000*n.sim_info.cost_time) if n.sim_info else 0
                tiu_param = null_regs
                alg_ops = 0
                arch_ops = 0
                if n.pmu_info.command:
                    alg_ops = n.pmu_info.command.alg_ops
                    arch_ops = n.pmu_info.command.arch_ops
                    tiu_param = get_tiu_param(n.pmu_info.command, inst_header[reg_begin_index:])
                    if n.pmu_info.command.mlir_cmd and command_type in ["conv.normal", "conv.wrq", "conv.wrqrelu",
        "pord.depthwise", "pord.avgpooling", "pord.depthwiserelu", "pord.maxpooling",]:
                        reg = n.pmu_info.command.mlir_cmd.reg
                        conv_kh = max(conv_kh, reg.opd1_h)
                        conv_kw = max(conv_kw, reg.opd1_w)
                        conv_stride_h = max(conv_stride_h, reg.res_op_y_str)
                        conv_stride_w = max(conv_stride_w, reg.res_op_x_str)
                        conv_pad[0] = max(conv_pad[0], reg.opd0_up_pad)
                        conv_pad[1] = max(conv_pad[1], reg.opd0_dn_pad)
                        conv_pad[2] = max(conv_pad[2], reg.opd0_lf_pad)
                        conv_pad[3] = max(conv_pad[3], reg.opd0_rt_pad)

                layer_alg_ops += alg_ops
                layer_arch_ops += arch_ops
                item = [0, command_type, 0, "-", *get_inst_io_dtype(n.pmu_info.command), layer.layer_id,
                        arch_cycles,
                        n.pmu_info.inst_end_time - n.pmu_info.inst_start_time,
                        n.pmu_info.inst_start_time, n.pmu_info.inst_end_time,
                        inst_id, dep_id, str(n.pmu_info),
                        alg_ops, arch_ops, get_ratio_str(alg_ops, arch_ops),
                        *tiu_param]
                inst_data.append(item)

            load_bytes = 0
            load_time = 0
            store_bytes = 0
            store_time = 0
            for n in layer.gdma_nodes:
                if not n.pmu_info:
                    continue
                command_type  = n.pmu_info.command.type if n.pmu_info.command else "UNKNOWN"
                inst_id = f'GDMA-{n.pmu_info.inst_id}'
                dep_id = f'TIU-{n.pmu_info.command.dep_id if n.pmu_info.command else "UNKNOWN"}'
                m = n.pmu_info
                trans_bytes = m.d0_wr_bytes + m.d1_wr_bytes + m.gif_wr_bytes
                arch_cycles = int(1000*n.sim_info.cost_time) if n.sim_info else 0
                dma_param = get_dma_param(n.pmu_info.command, inst_header[reg_begin_index:])
                item = [2, command_type, trans_bytes, get_dma_direction(n.pmu_info.command),
                        *get_inst_io_dtype(n.pmu_info.command),
                        layer.layer_id,
                        arch_cycles,
                        n.pmu_info.inst_end_time - n.pmu_info.inst_start_time,
                        n.pmu_info.inst_start_time, n.pmu_info.inst_end_time,
                        inst_id, dep_id, str(n.pmu_info),
                        "-", "-", "-",
                        *dma_param]
                inst_data.append(item)
                if m.d0_wr_bytes + m.d1_wr_bytes > 0:
                    store_bytes += m.d0_wr_bytes + m.d1_wr_bytes
                    store_time += n.pmu_info.inst_end_time - n.pmu_info.inst_start_time
                if m.d0_ar_bytes + m.d1_ar_bytes > 0:
                    load_bytes += m.d0_ar_bytes + m.d1_ar_bytes
                    load_time += n.pmu_info.inst_end_time - n.pmu_info.inst_start_time

            if layer.layer_id == -1:
                continue
            if layer.begin_usec is None:
                # print("WARNING: layer_id={} layer_type={} has no time info".format(layer.layer_id, layer_type))
                continue
            if len(layer.bd_nodes) == 0:
                continue
            tiu_cycles = sum([n.pmu_info.inst_end_time - n.pmu_info.inst_start_time if n.pmu_info else 0 for n in layer.bd_nodes])
            if tiu_cycles == 0:
                continue
            arch_cycles = sum(
                int(1000 * n.sim_info.cost_time) if n.sim_info else 0
                for n in layer.bd_nodes
            )
            total_tiu_cycles += tiu_cycles
            total_arch_cycles += arch_cycles
            total_alg_ops += layer_alg_ops
            total_arch_ops += layer_arch_ops

            load_bandwidth = load_bytes/(1024*1024*1024)/(load_time*GDMAPeriod*1e-6) if load_time > 0 else 0
            store_bandwidth = store_bytes/(1024*1024*1024)/(store_time*GDMAPeriod*1e-6) if store_time > 0 else 0

            if layer.layer_id not in layer_id_map:
                item = [layer.layer_id,
                        "local" if layer.is_local else "global",
                        "TPU",
                        get_layer_dtype(layer),
                        layer_type,
                        load_bandwidth, store_bandwidth,
                        conv_kh, conv_kw, conv_stride_h, conv_stride_w, str(conv_pad),
                        *get_layer_nchw(layer, True), *get_layer_nchw(layer, False),
                        get_layer_bytes(layer, True), get_layer_bytes(layer, False), get_layer_weight_size(layer),  #"ifm(B)", "ofm(B)", "Weight(B)"
                        layer_alg_ops, layer.io_info(), #"AlgorithmOps", "Other info",
                        0, layer_arch_ops, arch_cycles, 0, #"uArch URate", "uArch Ops", "uArch Cycle", "uArch Cycle Ratio",
                        tiu_cycles, 0]
                layer_id_map[layer.layer_id] = item
            else:
                layer_id_map[layer.layer_id][3] |= get_layer_dtype(layer)
                layer_id_map[layer.layer_id][-2] += tiu_cycles
                layer_id_map[layer.layer_id][-4] += arch_cycles
                layer_id_map[layer.layer_id][-5] += layer_arch_ops
                layer_id_map[layer.layer_id][-8] += layer_alg_ops

        layer_data = layer_id_map.values()
        layer_data = sorted(layer_data, key=lambda x: x[-2], reverse=True)
        for layer in layer_data:
            layer[-1] = get_ratio_str(layer[-2], total_tiu_cycles)
            layer[-3] = get_ratio_str(layer[-4], total_arch_cycles)
            layer[-6] = get_ratio_str(layer[-8], layer[-5])

        layer_file = os.path.join(out_dir, "layer.csv")
        inst_file = os.path.join(out_dir, "instruction.csv")
        summary_file = os.path.join(out_dir, "summary.csv")

        layer_summary_map = dict()
        layer_summary_header = ["Function", "Algorithm Ops", "Algorithm Tops Ratio", "Weight Size", "uArch Tops", "uArch URate", "uArch Tops Ratio", "1684x Cycles", "1684x Ratio", "1684x FPS or Token/s", "Layer Types"]
        layer_summary_row_map = {
                              "conv":"Conv",
                              "conv2d":"Conv",
                              "pool":"Pool",
                              "pool2d":"Pool",
                              "fc":"Matmul",
                              "matmul":"Matmul",
                              "batch_matmul":"Matmul",
                              "softmax":"Softmax",
                              "batchnorm":"BatchNorm",
                              "layernorm":"LayerNorm",
                              "layer_norm":"LayerNorm",
                              "group_norm":"LayerNorm",
                              "groupnorm":"LayerNorm",
                              "eltwise":"Eltwise",
                              "eltwise_binary":"Eltwise",
                              "mul":"Eltwise",
                              "add":"Eltwise",
                              "broadcast_binary":"Eltwise",
                              }
        total_weight_size = 0
        for layer in layer_data:
            layer_type = layer[4]
            row_name = layer_summary_row_map.get(layer_type.lower(), "Others")
            layer_alg_ops = layer[-8]
            layer_arch_ops = layer[-5]
            weight_size = layer[-9]
            layer_cycles = layer[-2]
            total_weight_size += weight_size
            item = [row_name, layer_alg_ops, 0, weight_size, layer_arch_ops, 0, 0, layer_cycles, 0, 0, {layer_type}]
            if row_name not in layer_summary_map:
                layer_summary_map[row_name] = item
            else:
                for i in range(1, len(item)-1):
                    layer_summary_map[row_name][i] += item[i]
                layer_summary_map[row_name][-1] |= item[-1]

        layer_summary_data = []
        layer_summary_rows = [
            "Conv",
            "Pool",
            "Matmul",
            "Softmax",
            "BatchNorm",
            "LayerNorm",
            "Eltwise",
            "Others",
        ]
        null_row = [0]*(len(layer_summary_header)-2) + [set()]
        for row in layer_summary_rows:
            layer_summary_data.append(layer_summary_map.get(row, [row] + null_row))

        for summary in layer_summary_data:
            summary[2] = get_ratio_str(summary[1], total_alg_ops)
            summary[6] = get_ratio_str(summary[4], total_arch_ops)
            summary[5] = get_ratio_str(summary[1], summary[4])
            summary[8] = get_ratio_str(summary[7], total_tiu_cycles)
            summary[-1] = ",".join(summary[-1])
        total_arch_urate = get_ratio_str(total_alg_ops, total_arch_ops)
        layer_summary_data.append(["Overall", total_alg_ops, "100%", total_weight_size, total_arch_ops, total_arch_urate, "100%", total_tiu_cycles, "100%", 1e6/(total_tiu_cycles*BDPeriod) if total_tiu_cycles != 0 else "--", ""])
        self.__write_csv(inst_file, inst_header, inst_data)
        if layer_data:
            self.__write_csv(layer_file, layer_header, layer_data)
            self.__write_csv(summary_file, layer_summary_header, layer_summary_data)
        else:
            print("No static layer info found!")

'''
        bmlib_inst_file = os.path.join(out_dir, "bmlib_instruction.csv")
        bmlib_inst_header = ["layer_id", "layer_type", "cmd_type", "cmd_id", "cycle", "start cycle", "end cycle", "params"]
        bmlib_inst_data = []

        bmlib_layers = []
        bmlib_layer_file = os.path.join(out_dir, "bmlib_layer.csv")
        bmlib_layer_header = ["layer_id", "layer_type", "cycles", "gdma number", "gdma cycles", "tiu number", "tiu cycles", "start_cycle", "end cycle"]
        bmlib_layer_data = []
        for item in bmlib_data:
            all_nodes = item.monitor_bd + item.monitor_gdma
            if all_nodes:
                all_nodes = sorted(all_nodes, key = lambda n: n.inst_start_time)
                for n in all_nodes:
                    node_type = "TIU" if type(n) == archlib.BDProfileFormat else "GDMA"
                    layer_type = ""
                    layer_id = item.summary.iteration
                    if n.dynamic:
                        d = n.dynamic
                        if hasattr(d, "layer"):
                            layer = d.layer
                            layer_type = layer.info
                            layer_id = layer_id + "-" + str(d.layer.layer_id)
                            if layer not in bmlib_layers:
                                layer.layer_type = layer_type
                                layer.layer_id = layer_id
                                bmlib_layers.append(layer)
                        elif hasattr(d, "wait"):
                            layer = d.wait
                            layer_type = "unknown"
                            layer_id ="wait" + str(d.wait.end_usec)
                            if layer not in bmlib_layers:
                                layer.layer_type = layer_type
                                layer.layer_id = layer_id
                                bmlib_layers.append(layer)
                    bmlib_inst_data.append([layer_id, layer_type, node_type, n.inst_id,
                                            n.inst_end_time - n.inst_start_time,
                                            n.inst_start_time, n.inst_end_time,
                                            get_inst_param(n.command) if n.command else ""])

        for layer in bmlib_layers:
            layer_nodes = layer.gdma_nodes + layer.bd_nodes
            print(len(layer_nodes))
            start_cycle = min([100000000] + [n.pmu_info.inst_start_time for n in layer_nodes if n.pmu_info])
            end_cycle = max([0] + [n.pmu_info.inst_end_time for n in layer_nodes if n.pmu_info])
            item = [layer.layer_id, layer.info,
                    end_cycle - start_cycle,
                    len(layer.gdma_nodes), sum([0] + [n.pmu_info.inst_end_time - n.pmu_info.inst_start_time for n in layer.gdma_nodes if n.pmu_info]),
                    len(layer.bd_nodes), sum([0] + [n.pmu_info.inst_end_time - n.pmu_info.inst_start_time for n in layer.bd_nodes if n.pmu_info]),
                    start_cycle, end_cycle,
                    ]
            bmlib_layer_data.append(item)
        self.__write_csv(bmlib_inst_file, bmlib_inst_header, bmlib_inst_data)
        self.__write_csv(bmlib_layer_file, bmlib_layer_header, bmlib_layer_data)

'''
