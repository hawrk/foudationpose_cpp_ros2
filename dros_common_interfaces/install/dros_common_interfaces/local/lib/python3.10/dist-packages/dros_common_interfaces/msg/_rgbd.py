# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dros_common_interfaces:msg/RGBD.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RGBD(type):
    """Metaclass of message 'RGBD'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dros_common_interfaces')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dros_common_interfaces.msg.RGBD')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__rgbd
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__rgbd
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__rgbd
            cls._TYPE_SUPPORT = module.type_support_msg__msg__rgbd
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__rgbd

            from sensor_msgs.msg import CameraInfo
            if CameraInfo.__class__._TYPE_SUPPORT is None:
                CameraInfo.__class__.__import_type_support__()

            from sensor_msgs.msg import Image
            if Image.__class__._TYPE_SUPPORT is None:
                Image.__class__.__import_type_support__()

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RGBD(metaclass=Metaclass_RGBD):
    """Message class 'RGBD'."""

    __slots__ = [
        '_header',
        '_rgb_camera_info',
        '_depth_camera_info',
        '_rgb',
        '_depth',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'rgb_camera_info': 'sensor_msgs/CameraInfo',
        'depth_camera_info': 'sensor_msgs/CameraInfo',
        'rgb': 'sensor_msgs/Image',
        'depth': 'sensor_msgs/Image',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'CameraInfo'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'CameraInfo'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'Image'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'Image'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        from sensor_msgs.msg import CameraInfo
        self.rgb_camera_info = kwargs.get('rgb_camera_info', CameraInfo())
        from sensor_msgs.msg import CameraInfo
        self.depth_camera_info = kwargs.get('depth_camera_info', CameraInfo())
        from sensor_msgs.msg import Image
        self.rgb = kwargs.get('rgb', Image())
        from sensor_msgs.msg import Image
        self.depth = kwargs.get('depth', Image())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.rgb_camera_info != other.rgb_camera_info:
            return False
        if self.depth_camera_info != other.depth_camera_info:
            return False
        if self.rgb != other.rgb:
            return False
        if self.depth != other.depth:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def rgb_camera_info(self):
        """Message field 'rgb_camera_info'."""
        return self._rgb_camera_info

    @rgb_camera_info.setter
    def rgb_camera_info(self, value):
        if __debug__:
            from sensor_msgs.msg import CameraInfo
            assert \
                isinstance(value, CameraInfo), \
                "The 'rgb_camera_info' field must be a sub message of type 'CameraInfo'"
        self._rgb_camera_info = value

    @builtins.property
    def depth_camera_info(self):
        """Message field 'depth_camera_info'."""
        return self._depth_camera_info

    @depth_camera_info.setter
    def depth_camera_info(self, value):
        if __debug__:
            from sensor_msgs.msg import CameraInfo
            assert \
                isinstance(value, CameraInfo), \
                "The 'depth_camera_info' field must be a sub message of type 'CameraInfo'"
        self._depth_camera_info = value

    @builtins.property
    def rgb(self):
        """Message field 'rgb'."""
        return self._rgb

    @rgb.setter
    def rgb(self, value):
        if __debug__:
            from sensor_msgs.msg import Image
            assert \
                isinstance(value, Image), \
                "The 'rgb' field must be a sub message of type 'Image'"
        self._rgb = value

    @builtins.property
    def depth(self):
        """Message field 'depth'."""
        return self._depth

    @depth.setter
    def depth(self, value):
        if __debug__:
            from sensor_msgs.msg import Image
            assert \
                isinstance(value, Image), \
                "The 'depth' field must be a sub message of type 'Image'"
        self._depth = value
