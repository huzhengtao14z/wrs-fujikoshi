# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: baxter_server.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='baxter_server.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x13\x62\x61xter_server.proto\"3\n\x13Gripper_pos_armname\x12\x0b\n\x03pos\x18\x01 \x01(\x02\x12\x0f\n\x07\x61rmname\x18\x02 \x01(\t\"\x1a\n\x07\x41rmname\x12\x0f\n\x07\x61rmname\x18\x01 \x01(\t\"\x1a\n\x0bGripper_pos\x12\x0b\n\x03pos\x18\x01 \x01(\x02\" \n\nJnt_angles\x12\x12\n\njnt_angles\x18\x01 \x01(\x0c\"H\n\x12Jnt_angles_armname\x12\x12\n\njnt_angles\x18\x01 \x01(\x0c\x12\r\n\x05speed\x18\x02 \x01(\x02\x12\x0f\n\x07\x61rmname\x18\x03 \x01(\t\"\x18\n\x08ListData\x12\x0c\n\x04list\x18\x01 \x01(\x0c\"\x1b\n\x0b\x43\x61mera_name\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x07\n\x05\x45mpty2\xce\x02\n\x0c\x42\x61xterServer\x12\x31\n\x0f\x62xt_set_gripper\x12\x14.Gripper_pos_armname\x1a\x06.Empty\"\x00\x12+\n\x0f\x62xt_get_gripper\x12\x08.Armname\x1a\x0c.Gripper_pos\"\x00\x12\'\n\x0c\x62xt_get_jnts\x12\x08.Armname\x1a\x0b.Jnt_angles\"\x00\x12\x32\n\x11\x62xt_movejnts_cont\x12\x13.Jnt_angles_armname\x1a\x06.Empty\"\x00\x12-\n\x0c\x62xt_movejnts\x12\x13.Jnt_angles_armname\x1a\x06.Empty\"\x00\x12&\n\rbxt_get_force\x12\x08.Armname\x1a\t.ListData\"\x00\x12*\n\rbxt_get_image\x12\x0c.Camera_name\x1a\t.ListData\"\x00\x62\x06proto3')
)




_GRIPPER_POS_ARMNAME = _descriptor.Descriptor(
  name='Gripper_pos_armname',
  full_name='Gripper_pos_armname',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pos', full_name='Gripper_pos_armname.pos', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='armname', full_name='Gripper_pos_armname.armname', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=23,
  serialized_end=74,
)


_ARMNAME = _descriptor.Descriptor(
  name='Armname',
  full_name='Armname',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='armname', full_name='Armname.armname', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=76,
  serialized_end=102,
)


_GRIPPER_POS = _descriptor.Descriptor(
  name='Gripper_pos',
  full_name='Gripper_pos',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pos', full_name='Gripper_pos.pos', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=104,
  serialized_end=130,
)


_JNT_ANGLES = _descriptor.Descriptor(
  name='Jnt_angles',
  full_name='Jnt_angles',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='jnt_angles', full_name='Jnt_angles.jnt_angles', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=132,
  serialized_end=164,
)


_JNT_ANGLES_ARMNAME = _descriptor.Descriptor(
  name='Jnt_angles_armname',
  full_name='Jnt_angles_armname',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='jnt_angles', full_name='Jnt_angles_armname.jnt_angles', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='speed', full_name='Jnt_angles_armname.speed', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='armname', full_name='Jnt_angles_armname.armname', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=166,
  serialized_end=238,
)


_LISTDATA = _descriptor.Descriptor(
  name='ListData',
  full_name='ListData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='list', full_name='ListData.list', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=240,
  serialized_end=264,
)


_CAMERA_NAME = _descriptor.Descriptor(
  name='Camera_name',
  full_name='Camera_name',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='Camera_name.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=266,
  serialized_end=293,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=295,
  serialized_end=302,
)

DESCRIPTOR.message_types_by_name['Gripper_pos_armname'] = _GRIPPER_POS_ARMNAME
DESCRIPTOR.message_types_by_name['Armname'] = _ARMNAME
DESCRIPTOR.message_types_by_name['Gripper_pos'] = _GRIPPER_POS
DESCRIPTOR.message_types_by_name['Jnt_angles'] = _JNT_ANGLES
DESCRIPTOR.message_types_by_name['Jnt_angles_armname'] = _JNT_ANGLES_ARMNAME
DESCRIPTOR.message_types_by_name['ListData'] = _LISTDATA
DESCRIPTOR.message_types_by_name['Camera_name'] = _CAMERA_NAME
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Gripper_pos_armname = _reflection.GeneratedProtocolMessageType('Gripper_pos_armname', (_message.Message,), dict(
  DESCRIPTOR = _GRIPPER_POS_ARMNAME,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Gripper_pos_armname)
  ))
_sym_db.RegisterMessage(Gripper_pos_armname)

Armname = _reflection.GeneratedProtocolMessageType('Armname', (_message.Message,), dict(
  DESCRIPTOR = _ARMNAME,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Armname)
  ))
_sym_db.RegisterMessage(Armname)

Gripper_pos = _reflection.GeneratedProtocolMessageType('Gripper_pos', (_message.Message,), dict(
  DESCRIPTOR = _GRIPPER_POS,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Gripper_pos)
  ))
_sym_db.RegisterMessage(Gripper_pos)

Jnt_angles = _reflection.GeneratedProtocolMessageType('Jnt_angles', (_message.Message,), dict(
  DESCRIPTOR = _JNT_ANGLES,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Jnt_angles)
  ))
_sym_db.RegisterMessage(Jnt_angles)

Jnt_angles_armname = _reflection.GeneratedProtocolMessageType('Jnt_angles_armname', (_message.Message,), dict(
  DESCRIPTOR = _JNT_ANGLES_ARMNAME,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Jnt_angles_armname)
  ))
_sym_db.RegisterMessage(Jnt_angles_armname)

ListData = _reflection.GeneratedProtocolMessageType('ListData', (_message.Message,), dict(
  DESCRIPTOR = _LISTDATA,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:ListData)
  ))
_sym_db.RegisterMessage(ListData)

Camera_name = _reflection.GeneratedProtocolMessageType('Camera_name', (_message.Message,), dict(
  DESCRIPTOR = _CAMERA_NAME,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Camera_name)
  ))
_sym_db.RegisterMessage(Camera_name)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), dict(
  DESCRIPTOR = _EMPTY,
  __module__ = 'baxter_server_pb2'
  # @@protoc_insertion_point(class_scope:Empty)
  ))
_sym_db.RegisterMessage(Empty)



_BAXTERSERVER = _descriptor.ServiceDescriptor(
  name='BaxterServer',
  full_name='BaxterServer',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=305,
  serialized_end=639,
  methods=[
  _descriptor.MethodDescriptor(
    name='bxt_set_gripper',
    full_name='BaxterServer.bxt_set_gripper',
    index=0,
    containing_service=None,
    input_type=_GRIPPER_POS_ARMNAME,
    output_type=_EMPTY,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='bxt_get_gripper',
    full_name='BaxterServer.bxt_get_gripper',
    index=1,
    containing_service=None,
    input_type=_ARMNAME,
    output_type=_GRIPPER_POS,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='bxt_get_jnts',
    full_name='BaxterServer.bxt_get_jnts',
    index=2,
    containing_service=None,
    input_type=_ARMNAME,
    output_type=_JNT_ANGLES,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='bxt_movejnts_cont',
    full_name='BaxterServer.bxt_movejnts_cont',
    index=3,
    containing_service=None,
    input_type=_JNT_ANGLES_ARMNAME,
    output_type=_EMPTY,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='bxt_movejnts',
    full_name='BaxterServer.bxt_movejnts',
    index=4,
    containing_service=None,
    input_type=_JNT_ANGLES_ARMNAME,
    output_type=_EMPTY,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='bxt_get_force',
    full_name='BaxterServer.bxt_get_force',
    index=5,
    containing_service=None,
    input_type=_ARMNAME,
    output_type=_LISTDATA,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='bxt_get_image',
    full_name='BaxterServer.bxt_get_image',
    index=6,
    containing_service=None,
    input_type=_CAMERA_NAME,
    output_type=_LISTDATA,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_BAXTERSERVER)

DESCRIPTOR.services_by_name['BaxterServer'] = _BAXTERSERVER

# @@protoc_insertion_point(module_scope)
