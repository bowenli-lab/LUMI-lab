from opentrons import protocol_api

# Metadata defines the required parameters for the protocol to be uploaded and run on the OT-2.
metadata = {
    'protocolName': 'Shaker Demo Protocol',
    'author': 'Kuan Pang <kuan.pang@mail.utoronto.ca>',
    'description': 'Lipid synthesis to a 96-well plate',
}
requirements = {"robotType": "OT-2", "apiLevel": "2.15"}


def run(protocol: protocol_api.ProtocolContext):
    # shaker module
    hs_mod = protocol.load_module(module_name="heaterShakerModuleV1",
                                  location="10")
    # hs_mod.open_labware_latch()


    hs_mod.close_labware_latch()
    hs_mod.set_and_wait_for_shake_speed(300)
    protocol.delay(minutes=0.2)
    hs_mod.deactivate_shaker()
    hs_mod.open_labware_latch()

