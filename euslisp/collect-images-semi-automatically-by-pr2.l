#!/usr/bin/env roseus
;; -*- coding: utf-8 -*-

(ros::load-ros-package "jsk_gui_msgs")
(ros::load-ros-package "jsk_recognition_msgs")
(ros::load-ros-package "std_srvs")

(require "models/room73b2-scene.l")
(require "package://pr2eus/pr2-interface.l")


#|
Subscribing topics
  - ``~bbox`` (jsk_recognition_msgs/BoundingBox)

Calling services
  - ``~yes_no_button`` (jsk_gui_msgs/YesNo)
  - ``~save_request`` (std_srvs/Trigger)

Getting parameters
  - ``~show_irtviewer`` (bool)
  - ``~speak`` (bool)

Setting parameters
  - ``~scene_id`` (int)
  - ``~base_offset_x`` (int)
  - ``~base_offset_y`` (int)
  - ``~head_offset_p`` (int)
  - ``~head_offset_y`` (int)
|#


(defvar *show-irtviewer*)
(defvar *speak*)


(defun init ()
  (let ()
    (ros::roseus "collect_images_semi_automatically_by_pr2")

    (ros::ros-info "Initializing...")
    (setq *show-irtviewer* (ros::get-param "~show_irtviewer"))
    (setq *speak* (ros::get-param "~speak"))
    (unless (boundp '*ri*)
      (pr2-init nil))
    (when *show-irtviewer*
      (unless (boundp '*room*)
        (setq *room* (room73b2)))
      (send *room* :move-to
            (make-coords :pos (float-vector 0 6675 0)
                         :rpy (float-vector 0 0 0)) :world)
      (objects (list *pr2* *room*))
      )

    (pr2-tuckarm-pose)
    (update-*pr2*)
    (redraw)
    ))


(defun redraw ()
  (when *show-irtviewer*
    (send *irtviewer* :redraw)
    (send *irtviewer* :look-all
          (geo::make-bounding-box
           (flatten (send-all (send *pr2* :bodies) :vertices)))))
  )


(defun update-*pr2* ()
  (ros::spin-once)
  (send *pr2* :move-to (send *ri* :state :worldcoords) :world)
  (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
  )


(defun wait-for-user-input ()
  (let ((service-name "~yes_no_button")
        (req (instance jsk_gui_msgs::YesNoRequest :init))
        (res nil)
        (pressed-p nil))
    (ros::wait-for-service service-name)
    (ros::ros-info "Please select Yes to start program.")
    ;; Wait until Yes is pressed
    (while (not pressed-p)
      (setq res (ros::service-call service-name req))
      (when (send res :yes)
        (ros::ros-info "Requesting to start program.")
        (setq pressed-p t)))
    ))


(defun move-base-multi-view (ref-coords offset-x offset-y)
  (let* ((target-coords (send ref-coords :copy-worldcoords)))
    (setq target-coords
          (send target-coords :locate (float-vector offset-x offset-y 0)))
    (ros::ros-info "Moving base...")
    (ros::ros-info "Current coords: ~A" (send *ri* :state :worldcoords))
    (ros::ros-info "Target coords : ~A" (send target-coords :worldcoords))
    (when *speak*
      (speak-jp "動きます。注意してください。"))
    (unix:sleep 1)
    (send *ri* :move-to target-coords)
    (update-*pr2*)
    (redraw)
    ))


(defun move-head-multi-view (mirror-centroid offset-p-deg offset-y-deg)
  (let* ((target-coords (make-coords :pos mirror-centroid)))
    (ros::ros-info "Moving head...")
    (send *pr2* :head :look-at (send target-coords :worldpos))
    (send *pr2* :head :neck-p :joint-angle
          (+ (send *pr2* :head :neck-p :joint-angle) offset-p-deg))
    (send *pr2* :head :neck-y :joint-angle
          (+ (send *pr2* :head :neck-y :joint-angle) offset-y-deg))
    (ros::ros-info "Pitch angle of neck: ~A [degree]"
                   (send *pr2* :head :neck-p :joint-angle))
    (ros::ros-info "Yaw angle of neck  : ~A [degree]"
                   (send *pr2* :head :neck-y :joint-angle))
    (redraw)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (update-*pr2*)
    (redraw)
    ))


(defun recognize-bbox ()
  (let (bbox-msg centroid)
    (ros::ros-info "Recognizing bbox...")

    (while (null bbox-msg)
      ;; Subscribe bbox topic
      (ros::ros-info "Wait for subscribing bbox topic...")
      (setq bbox-msg
            (one-shot-subscribe "~bbox" jsk_recognition_msgs::BoundingBox))
      )

    ;; Get position of bbox (required frame: /map)
    ;; Convert metric: m -> mm
    (setq centroid
          (float-vector
           (* 1000 (send (send (send bbox-msg :pose) :position) :x))
           (* 1000 (send (send (send bbox-msg :pose) :position) :y))
           ;; For euslisp. /eng2/7f is 30000 [mm] height in euslisp model.
           (+ 30000 (* 1000 (send (send (send bbox-msg :pose) :position) :z)))))

    (when *speak*
      (speak-jp "注目すべき場所を教えてもらいました。"))
    (ros::ros-info "Bounding box found at frame /map : ~a" centroid)
    (return-from recognize-bbox centroid)
    ))


(defun main ()
  (let (cur-coords
        mirror-centroid
        (save-service-name "~save_request")
        (req (instance std_srvs::TriggerRequest :init))
        (res nil))
    (init)
    (while (ros::ok)
      (x::window-main-one)
      (when *speak*
        (speak-jp "画像データを集めます。")
        (speak-jp "前方に障害物がないことを確認したら、Yesボタンを押して開始してください。"))
      (wait-for-user-input)

      ;; Get current time
      (ros::set-param "~scene_id" (send (ros::time-now) :to-nsec))

      ;; Self relocalization
      (when *speak*
        (speak-jp "その場で旋回します。"))
      (dolist (i (list 0 1 2))
        (send *ri* :go-pos 0 0 120))
      (update-*pr2*)

      ;; Get current coords for reference
      (setq cur-coords (send (send *ri* :state :worldcoords) :copy-worldcoords))
      (ros::ros-info "current coords : ~A" cur-coords)

      ;; Subscribe bbox
      (setq mirror-centroid (recognize-bbox))

      (dolist (base-offset-x (list 0 500 1000 1500 2000))
        (dolist (base-offset-y (list 0))
          (ros::set-param "~base_offset_x" base-offset-x)
          (ros::set-param "~base_offset_y" base-offset-y)

          ;; Move base
          (move-base-multi-view cur-coords base-offset-x base-offset-y)

          (dolist (head-offset-p (list -20 0 20))
            (dolist (head-offset-y (list -20 0 20))
              (ros::set-param "~head_offset_p" head-offset-p)
              (ros::set-param "~head_offset_y" head-offset-y)

              ;; Move head
              (move-head-multi-view mirror-centroid head-offset-p head-offset-y)
              (unix:sleep 1)

              ;; Save request
              (ros::ros-info "Waiting for service: ~A" save-service-name)
              (ros::wait-for-service save-service-name)
              (ros::ros-info "Requesting to save images...")
              (setq res (ros::service-call save-service-name req))
              (when *speak*
                (if (send res :success)
                    (speak-jp "パシャ")
                  (speak-jp "撮影に失敗しました。")))
              (ros::ros-info "success: ~A, message: ~A"
                             (send res :success) (send res :message))
              (unix:sleep 1)
              )
            )
          )
        )
      )
    ))


(main)
