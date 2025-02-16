from opentrons import protocol_api

# Metadata defines the required parameters for the protocol to be uploaded and
# run on the OT-2.

metadata = {
    "protocolName": "Lipid Synthesis Protocol",
    "author": "Kuan Pang <kuan.pang@mail.utoronto.ca>",
    "description": "Lipid synthesis to a 96-well plate",
}
requirements = {"robotType": "OT-2", "apiLevel": "2.15"}

# targets: Dict[int, List[int]];
# Source well -> [Destination wells]
targets = {}


####################
# REPLACE ME WITH THE REST OF THE TARGETS
####################


def run(protocol: protocol_api.ProtocolContext):
    # Define labware
    single_tiprack = protocol.load_labware("opentrons_96_tiprack_300ul", "3")

    reagent_plate = protocol.load_labware(
        "thermoscientificnunc_96_wellplate_1300ul", "1"
    )
    destination_plate = protocol.load_labware("servicebio_96_wellplate_200ul", "2")

    # Define pipette
    p300 = protocol.load_instrument(
        "p300_single_gen2", "right", tip_racks=[single_tiprack]
    )

    # define the z-axis offset
    p300.well_bottom_clearance.dispense = 19

    # We distribute the reagents to the destination plate
    for source, target in targets.items():
        dest_wells = [destination_plate.wells()[dest] for dest in target]
        source_well = reagent_plate.wells()[source]
        p300.distribute(
            20,
            source=source_well,
            dest=dest_wells,
            new_tip="always",
            trash=False,
            blow_out=True,
            blowout_location="source well",
            disposal_volume=20,
        )
