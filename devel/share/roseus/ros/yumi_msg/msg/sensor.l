;; Auto-generated. Do not edit!


(when (boundp 'yumi_msg::sensor)
  (if (not (find-package "YUMI_MSG"))
    (make-package "YUMI_MSG"))
  (shadow 'sensor (find-package "YUMI_MSG")))
(unless (find-package "YUMI_MSG::SENSOR")
  (make-package "YUMI_MSG::SENSOR"))

(in-package "ROS")
;;//! \htmlinclude sensor.msg.html


(defclass yumi_msg::sensor
  :super ros::object
  :slots (_force_data ))

(defmethod yumi_msg::sensor
  (:init
   (&key
    ((:force_data __force_data) 0.0)
    )
   (send-super :init)
   (setq _force_data (float __force_data))
   self)
  (:force_data
   (&optional __force_data)
   (if __force_data (setq _force_data __force_data)) _force_data)
  (:serialization-length
   ()
   (+
    ;; float32 _force_data
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _force_data
       (sys::poke _force_data (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _force_data
     (setq _force_data (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get yumi_msg::sensor :md5sum-) "3673a37cbe0c6421479c4d377a481265")
(setf (get yumi_msg::sensor :datatype-) "yumi_msg/sensor")
(setf (get yumi_msg::sensor :definition-)
      "float32 force_data

")



(provide :yumi_msg/sensor "3673a37cbe0c6421479c4d377a481265")


