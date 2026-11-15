// Auto-generated. Do not edit!

// (in-package yumi_msg.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class sensor {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.force_data = null;
    }
    else {
      if (initObj.hasOwnProperty('force_data')) {
        this.force_data = initObj.force_data
      }
      else {
        this.force_data = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type sensor
    // Serialize message field [force_data]
    bufferOffset = _serializer.float32(obj.force_data, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type sensor
    let len;
    let data = new sensor(null);
    // Deserialize message field [force_data]
    data.force_data = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'yumi_msg/sensor';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '3673a37cbe0c6421479c4d377a481265';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 force_data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new sensor(null);
    if (msg.force_data !== undefined) {
      resolved.force_data = msg.force_data;
    }
    else {
      resolved.force_data = 0.0
    }

    return resolved;
    }
};

module.exports = sensor;
