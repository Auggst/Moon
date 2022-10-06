//#pragma once
//
//#include <memory>
//
//#include "camera.h"
//#include "hittable_list.h"
//
//namespace Moon {
//	using std::make_shared;
//	using std::shared_ptr;
//	class engine {
//	public:
//		shared_ptr<camera> cam;
//		shared_ptr<hittable_list> scene;
//	public:
//		engine();
//		engine(shared_ptr<camera> _cam, shared_ptr<hittable_list> _scene);
//		~engine();
//
//		static engine& getInstance() {
//			static engine value;
//			return value;
//		}
//
//		void init();
//		void update();
//		void onmousemove(int x, int y);
//		void onmousescroll(int a);
//	};
//}