from opentrons import protocol_api

metadata = {
    "protocolName": "Shaking Protocol",
    "author": "Kuan Pang <kuan.pang@mail.utoronto.ca>",
    "description": "Lipid shaking with 96-well plate",
}
requirements = {"robotType": "OT-2", "apiLevel": "2.15"}


def run(protocol: protocol_api.ProtocolContext):
    # shaker module
    hs_mod = protocol.load_module(module_name="heaterShakerModuleV1", location="7")
    # open the labware latch if it is closed
    if hs_mod.labware_latch_status not in ["opening", "idle_open"]:
        hs_mod.open_labware_latch()
    hs_mod.close_labware_latch()
    hs_mod.set_and_wait_for_shake_speed(400)
    # 18 hr -> 1080 min
    protocol.delay(minutes=1080)
    hs_mod.deactivate_shaker()
    hs_mod.open_labware_latch()
