# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: game_socket.proto
# Protobuf Python Version: 4.25.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11game_socket.proto\x12\x0bgame_socket\"\x8f\x04\n\rClientMessage\x12=\n\x13tank_control_update\x18\x01 \x01(\x0b\x32\x1e.game_socket.TankControlUpdateH\x00\x12@\n\x14subscription_request\x18\x02 \x01(\x0b\x32 .game_socket.SubscriptionRequestH\x00\x12;\n\x12spawn_tank_request\x18\x03 \x01(\x0b\x32\x1d.game_socket.SpawnTankRequestH\x00\x12;\n\x12tanks_list_request\x18\x04 \x01(\x0b\x32\x1d.game_socket.TanksListRequestH\x00\x12>\n\x13observation_request\x18\x05 \x01(\x0b\x32\x1f.game_socket.ObservationRequestH\x00\x12\x39\n\x11kill_tank_request\x18\x06 \x01(\x0b\x32\x1c.game_socket.KillTankRequestH\x00\x12\x41\n\x15turret_control_update\x18\x07 \x01(\x0b\x32 .game_socket.TurretControlUpdateH\x00\x12:\n\x11\x62\x61ll_list_request\x18\x08 \x01(\x0b\x32\x1d.game_socket.BallsListRequestH\x00\x42\t\n\x07message\"\x12\n\x10SpawnTankRequest\"\"\n\x0fKillTankRequest\x12\x0f\n\x07tank_id\x18\x01 \x01(\x04\"\x12\n\x10TanksListRequest\"\x12\n\x10\x42\x61llsListRequest\"U\n\x11TankControlUpdate\x12\x0f\n\x07tank_id\x18\x01 \x01(\x04\x12/\n\x08\x63ontrols\x18\x02 \x01(\x0b\x32\x1d.game_socket.TankControlState\"=\n\x10TankControlState\x12\x14\n\x0cright_engine\x18\x01 \x01(\x02\x12\x13\n\x0bleft_engine\x18\x02 \x01(\x02\"[\n\x13TurretControlUpdate\x12\x11\n\tturret_id\x18\x01 \x01(\x04\x12\x31\n\x08\x63ontrols\x18\x02 \x01(\x0b\x32\x1f.game_socket.TurretControlState\";\n\x12TurretControlState\x12\x16\n\x0erotation_speed\x18\x01 \x01(\x02\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\"\x81\x01\n\x13SubscriptionRequest\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\x04\x12\x36\n\x10observation_kind\x18\x02 \x01(\x0e\x32\x1c.game_socket.ObservationKind\x12\x15\n\x08\x63ooldown\x18\x03 \x01(\x02H\x00\x88\x01\x01\x42\x0b\n\t_cooldown\"\\\n\x12ObservationRequest\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\x04\x12\x36\n\x10observation_kind\x18\x02 \x01(\x0e\x32\x1c.game_socket.ObservationKind\"\x89\x02\n\rServerMessage\x12<\n\x12observation_update\x18\x01 \x01(\x0b\x32\x1e.game_socket.ObservationUpdateH\x00\x12)\n\x0ctank_spawned\x18\x02 \x01(\x0b\x32\x11.game_socket.TankH\x00\x12\x13\n\ttank_died\x18\x03 \x01(\x04H\x00\x12*\n\ttank_list\x18\x04 \x01(\x0b\x32\x15.game_socket.TankListH\x00\x12\x17\n\rtank_assigned\x18\x05 \x01(\x04H\x00\x12*\n\tball_list\x18\x06 \x01(\x0b\x32\x15.game_socket.BallListH\x00\x42\t\n\x07message\"\xf8\x02\n\x11ObservationUpdate\x12\x0e\n\x06\x65ntity\x18\x01 \x01(\x04\x12\x11\n\ttimestamp\x18\x02 \x01(\x04\x12#\n\x05image\x18\x03 \x01(\x0b\x32\x12.game_socket.ImageH\x00\x12\'\n\x07sensors\x18\x04 \x01(\x0b\x32\x14.game_socket.SensorsH\x00\x12\x36\n\rtank_controls\x18\x05 \x01(\x0b\x32\x1d.game_socket.TankControlStateH\x00\x12%\n\x06reward\x18\x06 \x01(\x0b\x32\x13.game_socket.RewardH\x00\x12:\n\x0fturret_controls\x18\x07 \x01(\x0b\x32\x1f.game_socket.TurretControlStateH\x00\x12)\n\x08position\x18\x08 \x01(\x0b\x32\x15.game_socket.PositionH\x00\x12\x1d\n\x13rotation_in_radians\x18\t \x01(\x02H\x00\x42\r\n\x0bobservation\"q\n\x05Image\x12.\n\traw_image\x18\x01 \x01(\x0b\x32\x19.game_socket.RawRgbaImageH\x00\x12*\n\tpng_image\x18\x02 \x01(\x0b\x32\x15.game_socket.PngImageH\x00\x42\x0c\n\nimage_type\";\n\x0cRawRgbaImage\x12\r\n\x05width\x18\x01 \x01(\r\x12\x0e\n\x06height\x18\x02 \x01(\r\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\"\x18\n\x08PngImage\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\"\t\n\x07Sensors\"(\n\x06Reward\x12\x0e\n\x06reward\x18\x01 \x01(\x01\x12\x0e\n\x06reason\x18\x02 \x01(\t\" \n\x08Position\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\",\n\x08TankList\x12 \n\x05tanks\x18\x01 \x03(\x0b\x32\x11.game_socket.Tank\"=\n\x04Tank\x12\x0f\n\x07tank_id\x18\x01 \x01(\x04\x12$\n\x07turrets\x18\x02 \x03(\x0b\x32\x13.game_socket.Turret\"\x1b\n\x06Turret\x12\x11\n\tturret_id\x18\x01 \x01(\x04\",\n\x08\x42\x61llList\x12 \n\x05\x62\x61lls\x18\x01 \x03(\x0b\x32\x11.game_socket.Ball\"\x17\n\x04\x42\x61ll\x12\x0f\n\x07\x62\x61ll_id\x18\x01 \x01(\x04*\x83\x01\n\x0fObservationKind\x12\x08\n\x04NONE\x10\x00\x12\t\n\x05IMAGE\x10\x01\x12\n\n\x06SENSOR\x10\x02\x12\x11\n\rTANK_CONTROLS\x10\x03\x12\x0b\n\x07REWARDS\x10\x04\x12\x13\n\x0fTURRET_CONTROLS\x10\x05\x12\x0c\n\x08POSITION\x10\x06\x12\x0c\n\x08ROTATION\x10\x07\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'game_socket_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_OBSERVATIONKIND']._serialized_start=2336
  _globals['_OBSERVATIONKIND']._serialized_end=2467
  _globals['_CLIENTMESSAGE']._serialized_start=35
  _globals['_CLIENTMESSAGE']._serialized_end=562
  _globals['_SPAWNTANKREQUEST']._serialized_start=564
  _globals['_SPAWNTANKREQUEST']._serialized_end=582
  _globals['_KILLTANKREQUEST']._serialized_start=584
  _globals['_KILLTANKREQUEST']._serialized_end=618
  _globals['_TANKSLISTREQUEST']._serialized_start=620
  _globals['_TANKSLISTREQUEST']._serialized_end=638
  _globals['_BALLSLISTREQUEST']._serialized_start=640
  _globals['_BALLSLISTREQUEST']._serialized_end=658
  _globals['_TANKCONTROLUPDATE']._serialized_start=660
  _globals['_TANKCONTROLUPDATE']._serialized_end=745
  _globals['_TANKCONTROLSTATE']._serialized_start=747
  _globals['_TANKCONTROLSTATE']._serialized_end=808
  _globals['_TURRETCONTROLUPDATE']._serialized_start=810
  _globals['_TURRETCONTROLUPDATE']._serialized_end=901
  _globals['_TURRETCONTROLSTATE']._serialized_start=903
  _globals['_TURRETCONTROLSTATE']._serialized_end=962
  _globals['_SUBSCRIPTIONREQUEST']._serialized_start=965
  _globals['_SUBSCRIPTIONREQUEST']._serialized_end=1094
  _globals['_OBSERVATIONREQUEST']._serialized_start=1096
  _globals['_OBSERVATIONREQUEST']._serialized_end=1188
  _globals['_SERVERMESSAGE']._serialized_start=1191
  _globals['_SERVERMESSAGE']._serialized_end=1456
  _globals['_OBSERVATIONUPDATE']._serialized_start=1459
  _globals['_OBSERVATIONUPDATE']._serialized_end=1835
  _globals['_IMAGE']._serialized_start=1837
  _globals['_IMAGE']._serialized_end=1950
  _globals['_RAWRGBAIMAGE']._serialized_start=1952
  _globals['_RAWRGBAIMAGE']._serialized_end=2011
  _globals['_PNGIMAGE']._serialized_start=2013
  _globals['_PNGIMAGE']._serialized_end=2037
  _globals['_SENSORS']._serialized_start=2039
  _globals['_SENSORS']._serialized_end=2048
  _globals['_REWARD']._serialized_start=2050
  _globals['_REWARD']._serialized_end=2090
  _globals['_POSITION']._serialized_start=2092
  _globals['_POSITION']._serialized_end=2124
  _globals['_TANKLIST']._serialized_start=2126
  _globals['_TANKLIST']._serialized_end=2170
  _globals['_TANK']._serialized_start=2172
  _globals['_TANK']._serialized_end=2233
  _globals['_TURRET']._serialized_start=2235
  _globals['_TURRET']._serialized_end=2262
  _globals['_BALLLIST']._serialized_start=2264
  _globals['_BALLLIST']._serialized_end=2308
  _globals['_BALL']._serialized_start=2310
  _globals['_BALL']._serialized_end=2333
# @@protoc_insertion_point(module_scope)