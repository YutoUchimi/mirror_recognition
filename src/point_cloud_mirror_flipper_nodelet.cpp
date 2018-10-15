/*****************************************************
 * Copyright (c) 2018, Yuto Uchimi
 * All rights reserved.
 ****************************************************/

#include "mirror_recognition/point_cloud_mirror_flipper.h"

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <jsk_recognition_utils/pcl_ros_util.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pluginlib/class_list_macros.h>


namespace mirror_recognition
{
  void PointCloudMirrorFlipper::onInit()
  {
    ConnectionBasedNodelet::onInit();
    pnh_->param("use_async", use_async_, false);
    pnh_->param("max_queue_size", max_queue_size_, 100);
    pub_ = advertise<sensor_msgs::PointCloud2>(*pnh_, "output", 1);
    onInitPostProcess();
  }

  void PointCloudMirrorFlipper::subscribe()
  {
    sub_input_.subscribe(*pnh_, "input", 1);
    sub_indices_.subscribe(*pnh_, "input/indices", 1);
    sub_coefficients_.subscribe(*pnh_, "input/coefficients", 1);

    if (use_async_)
    {
      async_ = boost::make_shared<message_filters::Synchronizer<ASyncPolicy> >(max_queue_size_);
      async_->connectInput(sub_input_, sub_indices_, sub_coefficients_);
      async_->registerCallback(boost::bind(&PointCloudMirrorFlipper::flip, this, _1, _2, _3));
    }
    else
    {
      sync_ = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(max_queue_size_);
      sync_->connectInput(sub_input_, sub_indices_, sub_coefficients_);
      sync_->registerCallback(boost::bind(&PointCloudMirrorFlipper::flip, this, _1, _2, _3));
    }
  }

  void PointCloudMirrorFlipper::unsubscribe()
  {
    sub_input_.unsubscribe();
    sub_indices_.unsubscribe();
    sub_coefficients_.unsubscribe();
  }

  void PointCloudMirrorFlipper::flip(const sensor_msgs::PointCloud2::ConstPtr& input,
                                     const jsk_recognition_msgs::ClusterPointIndices::ConstPtr& indices,
                                     const jsk_recognition_msgs::ModelCoefficientsArray::ConstPtr& coefficients)
  {
    boost::mutex::scoped_lock lock(mutex_);

    // Check frame_id
    if (!jsk_recognition_utils::isSameFrameId(input->header.frame_id, indices->header.frame_id) ||
        !jsk_recognition_utils::isSameFrameId(input->header.frame_id, coefficients->header.frame_id))
    {
      NODELET_ERROR("frame_id does not match.");
      return;
    }

    // Convert PointCloud to the PCL types
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*input, *input_cloud);

    // Concatenate indices into one PointIndices, and
    // extract points outside mirror world from input cloud
    pcl::PointIndices::Ptr all_indices(new pcl::PointIndices);
    for (size_t i = 0; i < indices->cluster_indices.size(); i++)
    {
      std::vector<int> one_indices = indices->cluster_indices[i].indices;
      for (size_t j = 0; j < one_indices.size(); j++)
      {
        all_indices->indices.push_back(one_indices[j]);
      }
    }
    pcl::ExtractIndices<pcl::PointXYZRGB> extract_cloud_outside_mirror;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_outside_mirror(new pcl::PointCloud<pcl::PointXYZRGB>());
    extract_cloud_outside_mirror.setNegative(true);
    extract_cloud_outside_mirror.setKeepOrganized(false);
    extract_cloud_outside_mirror.setInputCloud(input_cloud);
    extract_cloud_outside_mirror.setIndices(all_indices);
    extract_cloud_outside_mirror.filter(*cloud_outside_mirror);

    // Flip points inside mirror world for each mirror plane
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr flipped_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (size_t i = 0; i < indices->cluster_indices.size(); i++)
    {
      // Extract points inside mirror world from input cloud
      pcl::PointIndices::Ptr indices_i(new pcl::PointIndices);
      for (size_t j = 0; j < indices->cluster_indices[i].indices.size(); j++)
      {
        indices_i->indices.push_back(indices->cluster_indices[i].indices[j]);
      }
      pcl::ExtractIndices<pcl::PointXYZRGB> extract_cloud_inside_one_mirror;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_inside_one_mirror(new pcl::PointCloud<pcl::PointXYZRGB>());
      extract_cloud_inside_one_mirror.setNegative(false);
      extract_cloud_inside_one_mirror.setKeepOrganized(false);
      extract_cloud_inside_one_mirror.setInputCloud(input_cloud);
      extract_cloud_inside_one_mirror.setIndices(indices_i);
      extract_cloud_inside_one_mirror.filter(*cloud_inside_one_mirror);

      // Each plane is expressed as [ax + by + cz + d = 0]
      float a, b, c, d;
      a = coefficients->coefficients[i].values[0];
      b = coefficients->coefficients[i].values[1];
      c = coefficients->coefficients[i].values[2];
      d = coefficients->coefficients[i].values[3];

      // Compute transform matrix
      Eigen::Matrix4f R;
      Eigen::Matrix4f R1 = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f R2 = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f R3 = Eigen::Matrix4f::Identity();
      if (c != 0)  // Translate plane along z-axis so that the plane contains origin
      {
        R1(2, 3) = d / c;
        R3(2, 3) = -d / c;
      }
      Eigen::Vector4f u2(a, b, c, 0);
      u2 = u2.normalized();
      R2 = R2 - 2 * u2 * u2.transpose();
      R = R3 * R2 * R1;

      // Do flip each cloud by mirror and concatenate them
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr flipped_one_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      pcl::transformPointCloud(*cloud_inside_one_mirror, *flipped_one_cloud, R);
      pcl::concatenateFields(*flipped_cloud, *flipped_one_cloud, *flipped_cloud);
    }

    // Concatenate all result
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::concatenateFields(*cloud_outside_mirror, *flipped_cloud, *output_cloud);

    // Publish PointCloud
    sensor_msgs::PointCloud2 ros_output;
    pcl::toROSMsg(*output_cloud, ros_output);
    ros_output.header = input->header;
    pub_.publish(ros_output);
  }
}  // namespace mirror_recognition

PLUGINLIB_EXPORT_CLASS(mirror_recognition::PointCloudMirrorFlipper, nodelet::Nodelet);
