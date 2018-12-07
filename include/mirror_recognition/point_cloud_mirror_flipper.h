/*****************************************************
 * Copyright (c) 2018, Yuto Uchimi
 * All rights reserved.
 ****************************************************/

#ifndef MIRROR_RECOGNITION_POINT_CLOUD_MIRROR_FLIPPER_H
#define MIRROR_RECOGNITION_POINT_CLOUD_MIRROR_FLIPPER_H

#include <jsk_recognition_msgs/ClusterPointIndices.h>
#include <jsk_recognition_msgs/ModelCoefficientsArray.h>
#include <jsk_topic_tools/connection_based_nodelet.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <ros/names.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace mirror_recognition
{
class PointCloudMirrorFlipper: public jsk_topic_tools::ConnectionBasedNodelet
{
public:
  typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::PointCloud2,
    jsk_recognition_msgs::ClusterPointIndices,
    jsk_recognition_msgs::ModelCoefficientsArray> SyncPolicy;
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::PointCloud2,
    jsk_recognition_msgs::ClusterPointIndices,
    jsk_recognition_msgs::ModelCoefficientsArray> ASyncPolicy;

protected:
  ////////////////////////////////////////////////
  // Methods
  ////////////////////////////////////////////////
  virtual void onInit();
  virtual void subscribe();
  virtual void unsubscribe();
  virtual void flip(const sensor_msgs::PointCloud2::ConstPtr& input,
                    const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& indices,
                    const jsk_recognition_msgs::ModelCoefficientsArray::ConstPtr& coefficients);

  ////////////////////////////////////////////////
  // ROS variables
  ////////////////////////////////////////////////
  boost::mutex mutex_;
  ros::Publisher pub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_input_;
  message_filters::Subscriber<jsk_recognition_msgs::ClusterPointIndices> sub_indices_;
  message_filters::Subscriber<jsk_recognition_msgs::ModelCoefficientsArray> sub_coefficients_;
  boost::shared_ptr<message_filters::Synchronizer<SyncPolicy> > sync_;
  boost::shared_ptr<message_filters::Synchronizer<ASyncPolicy> > async_;

  ////////////////////////////////////////////////
  // Parameters
  ////////////////////////////////////////////////
  bool approximate_sync_;
  int max_queue_size_;

private:
};

}  // namespace mirror_recognition

#endif  // MIRROR_RECOGNITION_POINT_CLOUD_MIRROR_FLIPPER_H
