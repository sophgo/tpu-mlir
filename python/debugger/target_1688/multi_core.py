# ==============================================================================
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

"""
This essentially simulates the engine synchronization mechanism of cmodel/hardware

Think of the instruction list for each core as a state machine.
The principle of consumption instructions:
if it is a normal instruction direct consumption, if it is a sys instruction,
it will be consumed according to the rules
1. The send command can be directly consumed
2. The wait command requires send_cnt to meet the requirements before it can be consumed
Assuming that a core instruction is in an unconsumable state
(wait is not up to the standard or all instructions have been consumed),
switch the current state machine

The sender sends a message to the message queue,
and the message queue adds 1 to the sent_cnt field
where the ID is located after receiving the message,
and sets the remain_wait_cnt field to wait_cnt;
the receiver polls whether the sent_cnt at the
corresponding position in the message queue has
met the requirements, if it is satisfied,
decrease reamin_wait_cnt by one.
When it is reduced to 0, this position is cleared to 0,
and the process of message synchronization is completed so far
"""
from enum import Enum
from .regdef import sDMA_sys_reg as dma_sys, SYS_reg as tiu_sys


class SYS_TYPE(Enum):
    SEND = 0
    WAIT = 1


class Status(Enum):
    # for msg use
    CONSUMED = 0
    WAITING = 1
    # for machine use
    RUNNING = 2
    MACHINE_DONE = 3


class CmdStateMachine:
    def __init__(self, cmds, msg_queue):
        self.cmds = cmds
        self.idx = 0
        self.msg_queue = msg_queue

    def consume_cmd(self):
        if self.consume_done():
            return None, Status.MACHINE_DONE
        cmd = self.cmds[self.idx]
        sys = (tiu_sys, dma_sys)
        if isinstance(cmd, sys):
            return self.consume_sys(cmd)
        else:
            self.idx += 1
        return cmd, Status.RUNNING

    def consume_sys(self, cmd):
        cmd, ret = self.msg_queue.consume_sys(cmd)
        if ret == Status.CONSUMED:
            self.idx += 1
            return cmd, Status.RUNNING
        else:
            return None, ret

    def consume_done(self):
        return self.idx >= len(self.cmds)


class Msg:
    def __init__(self):
        self.sent_cnt = 0
        self.wait_cnt = 0


class MsgQueue:
    def __init__(self):
        self.msges = [Msg()] * 128

    def get_cmd_type(self, cmd):
        if isinstance(cmd, tiu_sys):
            if cmd.tsk_eu_typ == 8:
                return SYS_TYPE.SEND
            elif cmd.tsk_eu_typ == 9:
                return SYS_TYPE.WAIT
            else:
                raise ValueError(f"cmd type error: {cmd}")
        elif isinstance(cmd, dma_sys):
            if cmd.cmd_special_function == 3:
                return SYS_TYPE.SEND
            elif cmd.cmd_special_function == 4:
                return SYS_TYPE.WAIT
            else:
                raise ValueError(f"cmd type error: {cmd}")
        else:
            raise ValueError(f"cmd type error: {cmd}")

    def get_msg_id(self, cmd):
        return cmd["msg_id"]

    def get_msg_cnt(self, cmd):
        return cmd["cnt"]

    def consume_sys(self, cmd):
        sys = (tiu_sys, dma_sys)
        assert isinstance(cmd, sys)
        if self.get_cmd_type(cmd) == SYS_TYPE.SEND:
            return self.consume_send(cmd)
        elif self.get_cmd_type(cmd) == SYS_TYPE.WAIT:
            return self.consume_wait(cmd)

    def consume_send(self, cmd):
        msg_id = self.get_msg_id(cmd)
        self.msges[msg_id].sent_cnt += 1
        self.msges[msg_id].remain_wait_cnt = self.get_msg_cnt(cmd)
        return cmd, Status.CONSUMED

    def consume_wait(self, cmd):
        msg_id = self.get_msg_id(cmd)
        if self.get_msg_cnt(cmd) != self.msges[msg_id].sent_cnt:
            return None, Status.WAITING
        else:
            self.msges[msg_id].remain_wait_cnt -= 1
            if self.msges[msg_id].remain_wait_cnt == 0:
                self.msges[msg_id].sent_cnt = 0
            return cmd, Status.CONSUMED


class MultiCoreCmd:
    def __init__(self, core_cmds):
        self.core_cmds = core_cmds
        self.core_num = len(core_cmds)
        self.cur_core_id = 0
        self.msg_queue = MsgQueue()
        self.cmd_state_machines = [
            CmdStateMachine(core_cmds[i], self.msg_queue) for i in range(self.core_num)
        ]
        self.current_machine = self.cmd_state_machines[self.cur_core_id]
        self.cmds = []

    def switch_machine(self):
        self.cur_core_id = (self.cur_core_id + 1) % self.core_num
        self.current_machine = self.cmd_state_machines[self.cur_core_id]

    def all_consume_done(self):
        for i in self.cmd_state_machines:
            if not i.consume_done():
                return False
        return True

    def consume_cmd(self):
        while not self.all_consume_done():
            cmd, ret = self.current_machine.consume_cmd()
            self.switch_machine()
            if ret not in (Status.MACHINE_DONE, Status.WAITING):
                self.cmds.append(cmd)
        return self.cmds
