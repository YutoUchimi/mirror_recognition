#!/usr/bin/env roseus

(require "package://pr2eus/pr2-interface.l")


(defun move-head-multi-view (view-frame-y view-frame-z mirror-centroid)
  (let ()
    (setq offset-y (+ -1000 (* 250 view-frame-y)))
    (setq offset-z (+ -1000 (* 250 view-frame-z)))
    (setq target-pos (float-vector (aref mirror-centroid 0)
                                   (+ (aref mirror-centroid 1) offset-y)
                                   (+ (aref mirror-centroid 2) offset-z)))
    (send *pr2* :head :look-at target-pos)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    ))


(defun main ()
  (setq *arm* :rarm)

  (jsk_arc2017_baxter::arc-init :ctype :rarm-head-controller :moveit nil)
  ;; (send *ti* :wipe-all-scene)

  ;; rotate head to opposite side of moving arm
  (send *baxter* :head_pan :joint-angle 70)
  (send *ri* :angle-vector-raw (send *baxter* :angle-vector) 3000 :head-controller 0)
  (send *ri* :wait-interpolation)

  (send *ti* :recognize-bin-boxes)
  (objects (append (list *baxter*) (send *ti* :visualize-bins)))

  ;; (send *ti* :add-cardboard-scene)
  ;; (send *ti* :add-bin-scene :b)
  ;; (send *ti* :add-bin-scene :c)

  (send *baxter* :fold-pose-back *arm*)
  (send *baxter* :head_pan :joint-angle 80)
  (send *ti* :send-av)

  ;; (send *ti* :delete-bin-scene :b)
  ;; (send *ti* :delete-bin-scene :c)

  (while (send *ti* :wait-for-user-input-to-start *arm*)
    (dolist (bin (list :b :c))
      (dolist (view-frame (list :1 :2 :3 :4 :5 :6 :7 :8 :9))
        ;; (ros::set-param "/right_hand/target_bin_name" (symbol2str bin))
        (cond ((eq bin :b) (ros::set-param "/right_hand/target_bin_name" "tort_back"))
              ((eq bin :c) (ros::set-param "/right_hand/target_bin_name" "tort_front")))
        (ros::set-param "/right_hand/view_frame" (symbol2str view-frame))
        ;; (send *ti* :add-bin-scene :b)
        ;; (send *ti* :add-bin-scene :c)
        ;; (send *ti* :move-arm-body->bin-overlook-pose *arm* bin)
        (move-arm-body->bin-overlook-pose->multi-view *arm* bin view-frame)
        ;; (send *baxter* *arm* :move-end-pos #f(0 0 100) :world)
        (send *ti* :send-av)
        (send *irtviewer* :redraw)
        (send *ri* :wait-interpolation)
        ;; (send *ti* :delete-bin-scene :b)
        ;; (send *ti* :delete-bin-scene :c)

        (unix::sleep 3)
        (setq res
              (ros::service-call "/right_hand/data_collection_server/save_request"
                                 (instance std_srvs::TriggerRequest :init)))
        (ros::ros-info-green "bin: ~A, view_frame: ~A, success: ~A, message: ~A" bin view-frame (send res :success) (send res :message))

        (send *baxter* :fold-pose-back *arm*)
        (send *ti* :send-av)
        )
      )
    )
  )


(main)
