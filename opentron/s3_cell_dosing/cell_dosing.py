from opentrons import protocol_api

# Metadata
metadata = {
    "protocolName": "cell dosing",
    "author": "Kuan Pang <kuan.pang@mail.utorronto.ca>",
    "description": "LNP cell dosing protocol",
}

# requirements
requirements = {"robotType": "OT-2", "apiLevel": "2.18"}

LNP_volume = 4  # uL


def run(protocol: protocol_api.ProtocolContext):

    # TODO: add the MC3 control

    # Labware Setup
    tiprack_20 = protocol.load_labware("opentrons_96_tiprack_20ul", str(4))
    cell_plate = protocol.load_labware("corning_96_wellplate_360ul_flat", "1")
    lnp_plate = protocol.load_labware("servicebio_96_wellplate_200ul", "2")

    # Pipette Setup
    p20_multi_pipette = protocol.load_instrument(
        "p20_multi_gen2",
        "left",
        tip_racks=[tiprack_20],
    )

    # Drop two tips for blank control
    tiprack_20.set_offset(x=0, y=54, z=0)
    p20_multi_pipette.pick_up_tip(tiprack_20.wells()[0])
    p20_multi_pipette.drop_tip()

    p20_multi_pipette.reset_tipracks()
    tiprack_20.set_offset(x=0, y=0, z=0)

    # Get the bottom of the wells
    lnp_plate_wells_bottom = [well.bottom(-2.8) for well in lnp_plate.wells()]
    cell_plate_wells_bottom = [well.bottom(-1) for well in cell_plate.wells()]

    # Transferring LNP into each corresponding well of cells
    p20_multi_pipette.transfer(
        LNP_volume,
        lnp_plate_wells_bottom,
        cell_plate_wells_bottom,
        new_tip="always",
        trash=False,
        mix_before=(1, 20),
        mix_after=(2, 20),
    )
