
"use strict";

let GetRobotInfo = require('./GetRobotInfo.js')
let SetRemoteLoggerLevel = require('./SetRemoteLoggerLevel.js')
let StopMotion = require('./StopMotion.js')
let CmdJointTrajectory = require('./CmdJointTrajectory.js')
let StartMotion = require('./StartMotion.js')
let SetDrivePower = require('./SetDrivePower.js')

module.exports = {
  GetRobotInfo: GetRobotInfo,
  SetRemoteLoggerLevel: SetRemoteLoggerLevel,
  StopMotion: StopMotion,
  CmdJointTrajectory: CmdJointTrajectory,
  StartMotion: StartMotion,
  SetDrivePower: SetDrivePower,
};
